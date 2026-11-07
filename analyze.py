import json,collections,numpy as np
from scipy.stats import wilcoxon,rankdata
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams['font.family']='DejaVu Sans'
import os
W=os.path.dirname(os.path.abspath(__file__))+os.sep
R=[json.loads(l) for l in open(W+'results.jsonl')]
M=['ce','gce','sce','apl','selc','smallloss','enn','cl','coteach','jocor','dividemix','dynec','dacon']
DS=['digits','mnist','fmnist','elec','weather','covtype']; NZ=['sym0.2','sym0.4','asym0.4','inst0.3']
A=collections.defaultdict(list); AUX=collections.defaultdict(list); T=collections.defaultdict(list)
for r in R:
    A[(r['ds'],r['noise'],r['method'])].append(r['acc']*100); T[(r['ds'],r['method'])].append(r['time'])
    if r['method'] in('dacon','dynec','smallloss','cl') and r.get('aux'): AUX[(r['ds'],r['noise'],r['method'])].append(r['aux'])
S={'main':{},'rank':{},'wil':{},'det':{},'time':{},'complete':[]}
for ds in DS:
  for nz in NZ:
    if all(len(A[(ds,nz,m)])==3 for m in M):
      S['complete'].append(f'{ds}|{nz}')
      for m in M: S['main'][f'{ds}|{nz}|{m}']=[float(np.mean(A[(ds,nz,m)])),float(np.std(A[(ds,nz,m)]))]
def ranks(group):
    keys=[k for k in S['complete'] if k.split('|')[0] in group]
    if not keys: return {}
    rk=np.array([rankdata([-S['main'][f'{k}|{m}'][0] for m in M]) for k in keys])
    return {m:float(rk[:,j].mean()) for j,m in enumerate(M)},len(keys)
IMG=['digits','mnist','fmnist']; TAB=['elec','weather','covtype']
for nm,g in [('image',IMG),('tabular',TAB),('all',DS)]:
    rr=ranks(g)
    if rr: S['rank'][nm]={'ranks':rr[0],'n':rr[1]}
for nm,g in [('image',IMG),('tabular',TAB)]:
    keys=[k for k in S['complete'] if k.split('|')[0] in g]
    if len(keys)>=5:
      for m in M[:-1]:
        d=[S['main'][f'{k}|dacon'][0]-S['main'][f'{k}|{m}'][0] for k in keys]
        try: p=float(wilcoxon(d).pvalue)
        except Exception: p=1.0
        S['wil'][f'{nm}|{m}']=[float(np.mean(d)),p,int(sum(x>0 for x in d)),len(d)]
for (ds,nz,m),v in AUX.items():
    ks=[k for k in v[0] if isinstance(v[0][k],(int,float))]
    S['det'][f'{ds}|{nz}|{m}']={k:float(np.mean([x[k] for x in v if k in x])) for k in ks}
for (ds,m),v in T.items(): S['time'][f'{ds}|{m}']=float(np.mean(v))
json.dump(S,open(W+'summary.json','w'),indent=1)
MN={'ce':'CE','gce':'GCE','sce':'SCE','apl':'APL','selc':'SELC','smallloss':'Small loss','enn':'ENN','cl':'CL','coteach':'Co-teaching','jocor':'JoCoR','dividemix':'DivideMix','dynec':'DyNeC','dacon':'DACON'}
if 'image' in S['rank'] and 'tabular' in S['rank']:
    fig,ax=plt.subplots(figsize=(10,3.6)); x=np.arange(len(M)); w=0.38
    ri=[S['rank']['image']['ranks'][m] for m in M]; rt=[S['rank']['tabular']['ranks'][m] for m in M]
    ax.bar(x-w/2,ri,w,label='Image benchmarks',color='#4c72b0',edgecolor='k',lw=.4,hatch='//')
    ax.bar(x+w/2,rt,w,label='Tabular benchmarks',color='#dd8452',edgecolor='k',lw=.4)
    ax.set_xticks(x); ax.set_xticklabels([MN[m] for m in M],rotation=35,ha='right',fontsize=9); ax.set_ylabel('Average rank (lower is better)',fontsize=9)
    ax.legend(fontsize=9); ax.grid(axis='y',alpha=.3); plt.tight_layout(); plt.savefig(W+'Fig3.png',dpi=300); plt.savefig(W+'Fig3.eps'); plt.close()
print(len(R),'runs; complete settings',len(S['complete']))
for k,v in S['rank'].items(): print(k,v['n'],sorted(v['ranks'].items(),key=lambda z:z[1])[:5])
import os
S['abl']={}
if os.path.exists(W+'ablation.jsonl'):
    AB=collections.defaultdict(list)
    for l in open(W+'ablation.jsonl'):
        r=json.loads(l); AB[(r['ds'],r['noise'],r['variant'])].append(r['acc']*100)
    for (ds,nz,v),x in AB.items():
        if len(x)==3: S['abl'][f'{ds}|{nz}|{v}']=float(np.mean(x))
    for k in list(S['abl']):
        ds,nz,v=k.split('|')
        for src,nm in [('dacon','full'),('dynec','no_dep')]:
            if len(A[(ds,nz,src)])==3: S['abl'][f'{ds}|{nz}|{nm}']=float(np.mean(A[(ds,nz,src)]))
json.dump(S,open(W+'summary.json','w'),indent=1)
