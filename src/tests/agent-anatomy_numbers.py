"""Numbers test for the agent-anatomy lab: every number the page shows, recomputed independently in Python.

The lab's runs are data (window.__ana.runDefs(): the task, the tool definitions' size, every tool call's name, input, result size and
error flag). This file rebuilds each run from that data with its own code:
- a block counts ceil(characters / 4) tokens; a tool_use block counts its JSON {"type":"tool_use","id":…,"name":…,"input":…};
- call i sends every block before it (system prompt, tool definitions, task, every earlier reply and tool result); the tokens read
  are the sum over calls;
- limits are checked before each call: calls made >= limit, or tokens read so far + this call's input > budget;
- time: a call takes t0 + input / prefill + output / decode; the tools of one reply run at the same time (the slowest counts).
It then checks the engine (__ana.build, chAll, chBest, drill) against it, the claims the copy makes, and the teaser (__tzAna).
Run: python3 tests/agent-anatomy_numbers.py"""
import json, math, pathlib, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/agent-anatomy.html', '--only', 'agent-anatomy'], check=True, cwd=P, stdout=subprocess.DEVNULL)
OUT = P / 'out' / 'agent-anatomy'; OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'wrapped-numbers.html').write_text('<!doctype html><html><head><meta charset="utf-8"></head><body>' + (P / 'out' / 'agent-anatomy.html').read_text() + '</body></html>')
fails, n = [], 0


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(name); print('FAIL', name, '::', str(info)[:300])


tok = lambda chars: -(-chars // 4)
TOOLT = {'read_file': 0.1, 'search': 0.3, 'edit_file': 0.1, 'run_lint': 1.2}


def run(defs, sys_chars, key, lim=0, budget=0, stop_on_err=False, pre=10000, dec=50, lat=0.5, tt=3):
    d = defs[key]
    blocks = [tok(sys_chars), tok(d['tools']['chars']), tok(len(d['task']))]
    steps = d['calls'] + ([None] if d['final'] else [])
    calls, used, uid, stop = [], 0, 0, None
    for st in steps:
        inp = sum(blocks)
        if lim and len(calls) >= lim: stop = 'limit'; break
        if budget and used + inp > budget: stop = 'budget'; break
        used += inp
        if st is None:
            out = tok(len(d['final'])); blocks.append(out)
            calls.append(dict(inp=inp, out=out, tm=lat + inp / pre + out / dec, tt=0)); stop = 'end'; break
        reply = []
        for c in st:
            uid += 1
            js = json.dumps({'type': 'tool_use', 'id': 'toolu_%02d' % uid, 'name': c['name'], 'input': c['input']}, separators=(',', ':'), ensure_ascii=False)
            reply.append(tok(len(js)))
        out = sum(reply); blocks += reply
        t_tools = max(tt if c['name'] == 'run_tests' else TOOLT[c['name']] for c in st)
        calls.append(dict(inp=inp, out=out, tm=lat + inp / pre + out / dec, tt=t_tools))
        blocks += [tok(c['chars']) for c in st]
        if stop_on_err and any(c['err'] for c in st): stop = 'error'; break
    stop = stop or 'end'
    return dict(n=len(calls), stop=stop, done=stop == 'end', ins=[c['inp'] for c in calls], total=sum(c['inp'] for c in calls),
                out=sum(c['out'] for c in calls), outs=[c['out'] for c in calls], tm=sum(c['tm'] for c in calls), tt=sum(c['tt'] for c in calls))


with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page()
    pg.goto('file://' + str(OUT / 'wrapped-numbers.html')); pg.wait_for_timeout(300)
    H = pg.evaluate("""()=>{const A=window.__ana, T=window.__tzAna; if(!A) return null;
      const opts=[{},{lim:5},{lim:10},{stopOnErr:true},{budget:60000},{pre:2000,dec:25,lat:1,tt:10},{pre:50000,dec:100,lat:0.2,tt:1}], B={};
      for(const k of ['base','par','terse','huge','long','chN','chS']) B[k]=opts.map(o=>A.build(k,o));
      return {defs:A.runDefs(), consts:A.consts, B, sends:A.sends('base','tests/test_cart.py'), chAll:A.chAll(), chBest:A.chBest(), drills:[1,2,3,4,5,6,7,8,9,10,11,12].map(s=>A.drill(s)), exits:A.exits(),
        tz:T?{calls:T.calls(),start:T.start(),sent:[0,1,2,3,4].map(i=>T.sent(i)),total:T.total()}:null};}""")
    b.close()

check('the hook is there', H is not None)
if H is None: sys.exit(1)
D, C = H['defs'], H['consts']
SYS = C['SYS']['chars']
OPTS = [{}, {'lim': 5}, {'lim': 10}, {'stop_on_err': True}, {'budget': 60000}, {'pre': 2000, 'dec': 25, 'lat': 1, 'tt': 10}, {'pre': 50000, 'dec': 100, 'lat': 0.2, 'tt': 1}]
close = lambda a, b: abs(a - b) < 1e-6
for k, runs in H['B'].items():
    for o, e in zip(OPTS, runs):
        mine = run(D, SYS, k, **o)
        same = (mine['n'], mine['stop'], mine['ins'], mine['total'], mine['out']) == (e['n'], e['stop'], [c['inTok'] for c in e['calls']], e['totalIn'], e['totalOut'])
        check(f'{k} {o}: calls, inputs, total, output tokens', same, (mine['n'], mine['stop'], mine['total'], e['n'], e['stop'], e['totalIn']))
        check(f'{k} {o}: model time and tool time', close(mine['tm'], e['tModel']) and close(mine['tt'], e['tTool']), (mine['tm'], e['tModel'], mine['tt'], e['tTool']))

base, par, terse, huge, long_ = (run(D, SYS, k) for k in ('base', 'par', 'terse', 'huge', 'long'))
# part 1 and part 3: 8 calls, the last about 12,000, the total in the right range and not in the wrong ones
check('base: 8 calls, inputs 4,227 … 11,882', base['n'] == 8 and base['ins'] == [4227, 5650, 9071, 9441, 9629, 11449, 11662, 11882], base['ins'])
check('base: 73,011 read in all, 6.1 times the last call', base['total'] == 73011 and f"{base['total'] / base['ins'][-1]:.1f}" == '6.1', base['total'])
check('part 1: “about 12,000” for the last call', round(base['ins'][-1] / 1000) * 1000 == 12000)
check('part 1: the answer is “30,000 to 90,000”; reading the last call once is under 30,000; every call at the last size is over 90,000',
      30000 <= base['total'] <= 90000 and base['ins'][-1] < 30000 and 8 * base['ins'][-1] > 90000 and C['PRED_RIGHT'] == 'mid')
check('part 6: the last call is “about a sixth” of the total', 0.15 < base['ins'][-1] / base['total'] < 0.18, base['ins'][-1] / base['total'])
check('part 3: call 8 resends 11,662 and adds 220', base['ins'][7] - base['ins'][6] == 220)
check('part 2: 8 replies, 7 tool calls', base['n'] == 8 and sum(len(s) for s in D['base']['calls']) == 7)
# part 3 (the owner, 2026-10-14): the bill counts each call's input plus its output; the replies are the output (the result and Show the math)
check('part 3: the bill: 73,011 input tokens and 242 output tokens; the replies write 23, 21, 20, 38, 20, 63, 20 and 37',
      base['total'] == 73011 and base['out'] == 242 and base['outs'] == [23, 21, 20, 38, 20, 63, 20, 37], (base['total'], base['out'], base['outs']))
check('part 3, Show the math: the input tokens are “about 300 times” the output (73,011 / 242 = 301.7); every reply is “a few dozen tokens” (20 to 63)',
      round(base['total'] / base['out'] / 10) * 10 == 300 and all(10 <= o < 100 for o in base['outs']), (base['total'] / base['out'], base['outs']))
# part 3, part 6 and Explain it, question 1: the model asks for the test file in call 1, so its content first goes out with call 2,
# and then with every later call: calls 2 to 8, 7 times (not 8)
asked = next(i + 1 for i, st in enumerate(D['base']['calls']) if any(c['name'] == 'read_file' and c['input'].get('path') == 'tests/test_cart.py' for c in st))
first, times = asked + 1, base['n'] - asked
check('part 3: the test file is asked for in call 1, first sent in call 2, sent 7 times in all', (asked, first, times) == (1, 2, 7), (asked, first, times))
check('part 3: the engine gives the same three counts', (H['sends']['asked'], H['sends']['first'], H['sends']['times']) == (asked, first, times), H['sends'])
check('part 3: call 1 sends no tool output (4,227 = system + tools + task); call 2 adds the test file', base['ins'][0] == tok(SYS) + tok(D['base']['tools']['chars']) + tok(len(D['base']['task']))
      and base['ins'][1] - base['ins'][0] > D['base']['calls'][0][0]['chars'] // 4, base['ins'][:2])
# part 4
stop = run(D, SYS, 'base', stop_on_err=True)
check('part 4: stop on error ends after call 5, not done', (stop['n'], stop['stop']) == (5, 'error'))
check('part 4: only “exit code 1” fixes it at call 10, two calls more, 18,109 more tokens', terse['n'] == 10 and terse['done'] and terse['total'] - base['total'] == 18109, terse['total'])
check('part 4: the test result after call 5 is the failure (is_error)', D['base']['calls'][4][0]['err'] and D['base']['calls'][4][0]['name'] == 'run_tests')
# part 5
check('part 5: three tools at once, 6 calls, 58,290 tokens', par['n'] == 6 and par['total'] == 58290, par)
slow_b, slow_p = run(D, SYS, 'base', tt=10), run(D, SYS, 'par', tt=10)
check('part 5: with a 10 s test run the tools take longer than the model calls (one per call: 20.7 s against 16.1 s; three at once: 20.5 s against 13.7 s)',
      slow_b['tt'] > slow_b['tm'] and slow_p['tt'] > slow_p['tm'] and f"{slow_b['tt']:.1f}" == '20.7' and f"{slow_b['tm']:.1f}" == '16.1' and f"{slow_p['tt']:.1f}" == '20.5',
      (slow_b['tt'], slow_b['tm'], slow_p['tt'], slow_p['tm']))
check('part 5: “in this lab’s assumptions” matters: at the initial speeds the model share is 71%, with a 10 s test run 44%',
      f"{100 * slow_b['tm'] / (slow_b['tm'] + slow_b['tt']):.0f}" == '44', slow_b['tm'] / (slow_b['tm'] + slow_b['tt']))
check('part 5: 22.8 s against 20.2 s; model calls 71% (one per call) and 68% (three at once)',
      f"{base['tm'] + base['tt']:.1f}" == '22.8' and f"{par['tm'] + par['tt']:.1f}" == '20.2' and f"{100 * base['tm'] / (base['tm'] + base['tt']):.0f}" == '71'
      and f"{100 * par['tm'] / (par['tm'] + par['tt']):.0f}" == '68', (base['tm'], base['tt'], par['tm'], par['tt']))
# scenarios
lim5 = run(D, SYS, 'base', lim=5)
check('scenario: a limit of 5 stops after call 5, right after the failing test', (lim5['n'], lim5['stop']) == (5, 'limit'))
check('scenario: 30 calls, the last 24,455, 447,828 in all, 18.3 times', long_['n'] == 30 and long_['ins'][-1] == 24455 and long_['total'] == 447828
      and f"{long_['total'] / long_['ins'][-1]:.1f}" == '18.3', (long_['n'], long_['ins'][-1], long_['total']))
check('scenario: the long log is 40,000 tokens; 50,090 in the last call; 187,635 in all', max(tok(c['chars']) for s in D['huge']['calls'] for c in s) == 40000
      and huge['ins'][-1] == 50090 and huge['total'] == 187635, (huge['ins'][-1], huge['total']))
check('scenarios: four presets', [x['id'] for x in C['PRESETS']] == ['lim', 'par', 'long', 'huge'])
# the challenge
nr = run(D, SYS, 'chN')
check('challenge: the normal run, 12 calls, 4,468 to 11,954, 117,258 in all', nr['n'] == 12 and nr['ins'][0] == 4468 and nr['ins'][-1] == 11954 and nr['total'] == 117258, nr)
grid = []
for l in C['CH']['lims']:
    for bud in C['CH']['buds']:
        a, s = run(D, SYS, 'chN', lim=l, budget=bud), run(D, SYS, 'chS', lim=l, budget=bud)
        grid.append(dict(lim=l, bud=bud, ok=a['done'], nCalls=a['n'], sCalls=s['n'], sStop=s['stop'], cost=s['total']))
check('challenge: 16 plans, as the engine computes them', [(g['lim'], g['bud'], g['ok'], g['nCalls'], g['sCalls'], g['sStop'], g['cost']) for g in grid]
      == [(e['lim'], e['bud'], e['ok'], e['nCalls'], e['sCalls'], e['sStop'], e['cost']) for e in H['chAll']])
ok = [g for g in grid if g['ok']]
best = min(g['cost'] for g in ok)
check('challenge: 6 plans let the normal run finish (limit 15 or more, budget 150,000 or more)', len(ok) == 6 and all(g['lim'] >= 15 and g['bud'] >= 150000 for g in ok))
check('challenge: the lowest cost is 138,675 (13 calls, stopped by the budget), shared by the three plans with budget 150,000',
      best == 138675 and sorted((g['lim'], g['bud']) for g in ok if g['cost'] == best) == [(15, 150000), (25, 150000), (50, 150000)]
      and all(g['sCalls'] == 13 and g['sStop'] == 'budget' for g in ok if g['cost'] == best))
check('challenge: chBest is limit 15, budget 150,000', (H['chBest']['lim'], H['chBest']['bud'], H['chBest']['cost']) == (15, 150000, 138675))
start = next(g for g in grid if (g['lim'], g['bud']) == (50, 300000))
check('challenge: the initial setup passes and lets the stuck run read 284,291 (21 calls)', start['ok'] and start['cost'] == 284291 and start['sCalls'] == 21)
check('challenge: a budget of 100,000 stops the normal run (it reads 117,258), and so does a limit of 10 (it needs 12)',
      all(not g['ok'] for g in grid if g['bud'] <= 100000 or g['lim'] == 10))
# the drill
for d in H['drills']:
    last = d['a'] + (d['n'] - 1) * d['g']; total = sum(d['a'] + i * d['g'] for i in range(d['n']))
    check(f"drill {d['seed']}: last {last}, total {total}", (d['last'], d['total']) == (last, total), d)
# exits: the two new cases
check('exits: 20 calls from 4,000 to 24,000 read about 280,000', sum(4000 + i * (24000 - 4000) / 19 for i in range(20)) == 280000 and H['exits']['transferA'][0] == 1)
check('exits: a 30,000-token log resent by 10 calls adds 300,000', 10 * 30000 == 300000 and H['exits']['transferA'][1] == 1)
# the teaser
T = H['tz']
check('teaser: the hook is there', T is not None)
if T:
    blocks, ins = [T['start']['you']], []
    for c in T['calls']:
        ins.append(sum(blocks)); blocks += [c['use'], c['res']]
    check('teaser: calls send 3,030 / 4,555 / 4,875 / 5,035 / 5,455', ins == [3030, 4555, 4875, 5035, 5455] and [s['all'] for s in T['sent']] == ins, ins)
    check('teaser: 22,950 in all, and not the lab’s numbers', sum(ins) == 22950 == T['total'] and sum(ins) != base['total'])

print(f'{n - len(fails)}/{n} passed')
sys.exit(1 if fails else 0)
