from engine import *
import sys
def mk(seed,n,rate,outr,mix,pre=8192,own=(200,600)):
    rnd=mulberry(seed); out=[]; t=0.0
    for i in range(n):
        t+=exp_gap(rnd,rate); u=rnd(); acc=0; app=mix[-1][0]
        for a,p in mix:
            acc+=p
            if u<acc: app=a; break
        out.append({'t':ms(t),'cls':'chat','app':app,'pre':pre,'own':uni(rnd,*own),'out':uni(rnd,*outr)})
    return out
MIX4=(('support',.4),('translate',.3),('writing',.2),('code',.1))
MIX6=(('a',.3),('b',.2),('c',.15),('d',.15),('e',.1),('f',.1))
for (n,rate,outr,mix,pre,cap) in [(48,4.0,(40,100),MIX4,8192,32),(48,6.0,(40,100),MIX4,8192,32),(48,4.0,(40,100),MIX6,8192,32),(48,3.0,(40,100),MIX6,12288,32),(40,4.0,(20,60),MIX6,8192,32)]:
    for seed in (21,22):
        ld=mk(seed,n,rate,outr,mix,pre)
        row=[]
        for name,ca,cb in (('off',False,None),('1',True,512),('2',True,1024),('4',True,2048),('6',True,3072),('inf',True,None)):
            if pre==12288 and cb: cb=cb*3//2
            run=simulate(ld,{'policy':'continuous','cap':cap,'budget':2048,'cache':ca,'cache_blocks':cb}); m=metrics(run,GOALS)
            row.append(f"{name}:{m['all']['ok']}/{m['all']['n']} p95 {m['chat']['ttft_p95']:.2f}")
        print(n,rate,outr,len(mix),pre,seed,'last %.1f'%ld[-1]['t'],' | '.join(row))
