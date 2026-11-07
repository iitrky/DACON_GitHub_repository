import json,os,time,sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import benchmarks as nl2
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),'results.jsonl'); done=set()
if os.path.exists(OUT):
    for l in open(OUT): done.add(json.loads(l)['key'])
DS=['digits','elec','weather','covtype','mnist','fmnist']
NZ=[('sym',0.2),('sym',0.4),('asym',0.4),('inst',0.3)]
M=['ce','gce','sce','apl','selc','smallloss','enn','cl','coteach','jocor','dividemix','dynec','dacon']
for ds in DS:
    Xtr,ytr,Xte,yte=nl2.load(ds); K=int(ytr.max()+1); c=nl2.cfg(ds,K,Xtr.shape[1])
    for kind,r in NZ:
        for sd in [0,1,2]:
            yn=nl2.corrupt(ytr,K,kind,r,sd,Xtr); D=(Xtr,yn,Xte,yte,ytr); er=float((yn!=ytr).mean())
            for m in M:
                key=f'{ds}|{kind}{r}|{sd}|{m}'
                if key in done: continue
                t=time.time(); a,x=nl2.run_method(m,D,K,c,sd,er)
                x={k:v for k,v in (x or {}).items() if k!='curve'}
                open(OUT,'a').write(json.dumps(dict(key=key,ds=ds,noise=f'{kind}{r}',seed=sd,method=m,acc=a,actual_noise=er,time=time.time()-t,aux=x))+'\n'); print(key,round(a,4),flush=True)
print('ALLDONE',flush=True)
