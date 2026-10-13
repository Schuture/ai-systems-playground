from engine import *
from l3probe import mk
MIX=(('manual',.5),('contract',.25),('codebase',.15),('paper',.1))
for pre in (24576, 32768):
  bpd = pre//16
  for (n,rate,outr) in [(48,1.5,(40,120)),(48,2.0,(40,120)),(60,2.5,(40,120)),(48,1.0,(40,120))]:
    for seed in (21,22,23):
        ld=mk(seed,n,rate,outr,MIX,pre,(100,400))
        row=[]
        for name,ca,cb in (('off',False,None),('1doc',True,bpd),('2doc',True,2*bpd),('4doc',True,4*bpd),('inf',True,None)):
            run=simulate(ld,{'policy':'continuous','cap':32,'budget':2048,'cache':ca,'cache_blocks':cb}); m=metrics(run,GOALS)
            hits=sum(1 for r in run['R'] if r.hit>=r.pre)
            row.append(f"{name}:{m['all']['ok']}/{m['all']['n']} hits {hits}")
        print(pre,n,rate,seed,'last %.1f'%ld[-1]['t'],' | '.join(row))
