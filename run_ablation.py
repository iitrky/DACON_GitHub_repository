import json,os,time,sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import benchmarks as nl2
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),'ablation.jsonl'); done=set()
if os.path.exists(OUT):
    for l in open(OUT): done.add(json.loads(l)['key'])
V={'no_dyn':{'dep':True,'dyn':False},'no_nbr':{'dep':True,'nbr':False},'no_relw':{'dep':True,'relw':False},'no_classcond':{'dep':True,'classcond':False},'no_correct':{'dep':True,'correct':False}}
import dacon_core as core
for ds,NZ in [('digits',[('sym',0.4),('asym',0.4),('inst',0.3)]),('mnist',[('asym',0.4),('inst',0.3)]),('fmnist',[('asym',0.4),('inst',0.3)])]:
    Xtr,ytr,Xte,yte=nl2.load(ds); K=int(ytr.max()+1); c=nl2.cfg(ds,K,Xtr.shape[1])
    for kind,r in NZ:
        for sd in [0,1,2]:
            yn=nl2.corrupt(ytr,K,kind,r,sd,Xtr); D=(Xtr,yn,Xte,yte,ytr)
            for n,fl in V.items():
                key=f'{ds}|{kind}{r}|{sd}|{n}'
                if key in done: continue
                a,aux=core.run(D,'dynec',c['epochs'],c['warm'],c['bs'],c['sizes'],sd,K,flags=fl,log_auc=True)
                open(OUT,'a').write(json.dumps(dict(key=key,ds=ds,noise=f'{kind}{r}',seed=sd,variant=n,acc=a,aux={k:v for k,v in aux.items() if k!='curve'}))+'\n'); print(key,a,flush=True)

