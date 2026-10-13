from engine import *
import itertools
res=[]
for rate in (4.0,5.0):
  for n in (48,):
    for outr in ((40,100),):
      for mix in ((('support',.4),('translate',.3),('writing',.2),('code',.1)),(('support',.55),('translate',.25),('writing',.12),('code',.08))):
        for seed in range(21,33):
          def ld_f(seed=seed):
            rnd=mulberry(seed); out=[]; t=0.0
            for i in range(n):
                t+=exp_gap(rnd,rate); u=rnd(); acc=0; app=mix[-1][0]
                for a,p in mix:
                    acc+=p
                    if u<acc: app=a; break
                out.append({'t':ms(t),'cls':'chat','app':app,'pre':8192,'own':uni(rnd,200,600),'out':uni(rnd,*outr)})
            return out
          ld=ld_f()
          if len(set(s['app'] for s in ld))<4: continue
          r={}
          for name,ca,cb in (('off',False,None),('1',True,512),('2',True,1024),('4',True,2048),('inf',True,None)):
              run=simulate(ld,{'policy':'continuous','cap':32,'budget':2048,'cache':ca,'cache_blocks':cb}); m=metrics(run,GOALS)
              r[name]=m['all']['ok_frac']
          if r['1']<0.9 and r['4']>=0.9 and r['inf']==r['4'] and r['off']<r['1']<=r['2']<=r['4'] and r['2']<r['4']+1e-9:
              res.append((round(r['off'],2),round(r['1'],2),round(r['2'],2),round(r['4'],2),rate,n,outr,mix[0][1],seed,ld[-1]['t']))
res.sort(key=lambda x:(-(x[3]-x[1]), x[9]))
for x in res[:25]: print(x)
print(len(res))
