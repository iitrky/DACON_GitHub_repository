"""Benchmarks, noise models and the reimplemented baselines (Co-teaching, JoCoR, DivideMix lite, SELC, confident learning, ENN)."""
import os, numpy as np, pandas as pd
import dacon_core as core
from dacon_core import MLP, softmax, gmm_clean
TAB={'covtype':'covtype.csv','elec':'elec.csv','weather':'weather.csv'}
def load(name):
    if name in ('mnist','fmnist','digits'):
        Xtr,ytr,Xte,yte=core.load(name,ntr=10000)
    else:
        d=pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),'data',TAB[name])); y=pd.factorize(d.iloc[:,-1],sort=True)[0]; X=d.iloc[:,:-1].values.astype(np.float32)
        r=np.random.RandomState(123).permutation(len(y))[:20000]; X,y=X[r],y[r]; n=14000
        Xtr,ytr,Xte,yte=X[:n],y[:n],X[n:],y[n:]
        m,s=Xtr.mean(0),Xtr.std(0)+1e-6; Xtr=(Xtr-m)/s; Xte=(Xte-m)/s
    return Xtr.astype(np.float32),ytr,Xte.astype(np.float32),yte
def corrupt(y,K,kind,rate,seed,X):
    if kind=='asym' and K==2:
        rng=np.random.RandomState(seed); yn=y.copy(); f=(rng.rand(len(y))<rate)&(y==0); yn[f]=1; return yn
    return core.corrupt(y,K,kind,rate,seed,X=X)
def cfg(name,K,d):
    if name=='digits': return dict(epochs=60,warm=10,bs=64,sizes=[d,128,64,K])
    if name in ('mnist','fmnist'): return dict(epochs=20,warm=5,bs=128,sizes=[d,256,128,K])
    return dict(epochs=20,warm=5,bs=128,sizes=[d,128,64,K])
def train_ce(Xtr,y,K,c,seed,epochs=None):
    rng=np.random.RandomState(seed); net=MLP(c['sizes'],rng); Y=np.eye(K,dtype=np.float32)[y]
    for ep in range(epochs or c['epochs']):
        perm=rng.permutation(len(y))
        for i in range(0,len(perm),c['bs']):
            b=perm[i:i+c['bs']]; z,A=net.fwd(Xtr[b]); p=softmax(z); net.bwd(A,((p-Y[b])/len(b)).astype(np.float32))
    return net
def acc(net,Xte,yte): return float((net.predict(Xte)[0].argmax(1)==yte).mean())
def sel_frac(ep,c,tau): return tau*min(1.0,ep/max(c['warm'],1))
def coteach(D,K,c,seed,tau,joint=False,lam=0.9):
    Xtr,yn,Xte,yte,yt=D; rng=np.random.RandomState(seed); A_=MLP(c['sizes'],rng); B_=MLP(c['sizes'],np.random.RandomState(seed+99)); Y=np.eye(K,dtype=np.float32)[yn]
    for ep in range(c['epochs']):
        perm=rng.permutation(len(yn)); fr=sel_frac(ep,c,tau)
        for i in range(0,len(perm),c['bs']):
            b=perm[i:i+c['bs']]; B=len(b); za,Aa=A_.fwd(Xtr[b]); zb,Ab=B_.fwd(Xtr[b]); pa=softmax(za); pb=softmax(zb)
            la=-np.log(pa[np.arange(B),yn[b]]+1e-8); lb=-np.log(pb[np.arange(B),yn[b]]+1e-8); keep=max(1,int(round((1-fr)*B)))
            if not joint:
                sa=np.argsort(la)[:keep]; sb=np.argsort(lb)[:keep]
                ga=np.zeros_like(pa); gb=np.zeros_like(pb); ga[sb]=pa[sb]-Y[b][sb]; gb[sa]=pb[sa]-Y[b][sa]
            else:
                lpa=np.log(pa+1e-8); lpb=np.log(pb+1e-8); kab=(pa*(lpa-lpb)).sum(1); kba=(pb*(lpb-lpa)).sum(1)
                jl=(1-lam)*(la+lb)+lam*(kab+kba); s=np.argsort(jl)[:keep]
                gA=(1-lam)*(pa-Y[b])+lam*((pa*(lpa-lpb)-pa*kab[:,None])+(pa-pb))
                gB=(1-lam)*(pb-Y[b])+lam*((pb*(lpb-lpa)-pb*kba[:,None])+(pb-pa))
                ga=np.zeros_like(pa); gb=np.zeros_like(pb); ga[s]=gA[s]; gb[s]=gB[s]
            A_.bwd(Aa,(ga/keep).astype(np.float32)); B_.bwd(Ab,(gb/keep).astype(np.float32))
    return acc(A_,Xte,yte)
def dividemix_lite(D,K,c,seed,T=0.5,alpha=4.0,lu=1.0):
    Xtr,yn,Xte,yte,yt=D; rng=np.random.RandomState(seed); nets=[MLP(c['sizes'],rng),MLP(c['sizes'],np.random.RandomState(seed+99))]; n=len(yn); Y=np.eye(K,dtype=np.float32)[yn]
    for ep in range(c['epochs']):
        if ep<c['warm']:
            for net in nets:
                perm=rng.permutation(n)
                for i in range(0,n,c['bs']):
                    b=perm[i:i+c['bs']]; z,A=net.fwd(Xtr[b]); net.bwd(A,((softmax(z)-Y[b])/len(b)).astype(np.float32))
            continue
        P=[net.predict(Xtr)[0] for net in nets]; W=[gmm_clean(-np.log(p[np.arange(n),yn]+1e-8),seed) for p in P]
        for k in (0,1):
            net=nets[k]; w=W[1-k]; lab=w>0.5; L_=np.where(lab)[0]; U=np.where(~lab)[0]
            if len(L_)<c['bs'] or len(U)<2: U=L_
            pk=P[k]; pavg=(P[0]+P[1])/2
            tl=w[:,None]*Y+(1-w[:,None])*pk; tl=tl**(1/T); tl/=tl.sum(1,keepdims=True)
            tu=pavg**(1/T); tu/=tu.sum(1,keepdims=True)
            permL=rng.permutation(L_)
            for i in range(0,len(permL),c['bs']):
                bl=permL[i:i+c['bs']]; bu=rng.choice(U,len(bl))
                Xb=np.vstack([Xtr[bl],Xtr[bu]]); Tb=np.vstack([tl[bl],tu[bu]]); wt=np.r_[np.ones(len(bl)),lu*np.ones(len(bu))]
                l=rng.beta(alpha,alpha); l=max(l,1-l); j=rng.permutation(len(Xb))
                Xm=l*Xb+(1-l)*Xb[j]; Tm=l*Tb+(1-l)*Tb[j]; wm=l*wt+(1-l)*wt[j]
                z,A=net.fwd(Xm.astype(np.float32)); net.bwd(A,((softmax(z)-Tm)*wm[:,None]/len(Xm)).astype(np.float32))
    pa=nets[0].predict(Xte)[0]+nets[1].predict(Xte)[0]; return float((pa.argmax(1)==yte).mean())
def selc(D,K,c,seed,alpha=0.9):
    Xtr,yn,Xte,yte,yt=D; rng=np.random.RandomState(seed); net=MLP(c['sizes'],rng); T=np.eye(K,dtype=np.float32)[yn].copy()
    for ep in range(c['epochs']):
        if ep>=c['warm']:
            P=net.predict(Xtr)[0]; T=(alpha*T+(1-alpha)*P).astype(np.float32)
        perm=rng.permutation(len(yn))
        for i in range(0,len(perm),c['bs']):
            b=perm[i:i+c['bs']]; z,A=net.fwd(Xtr[b]); net.bwd(A,((softmax(z)-T[b])/len(b)).astype(np.float32))
    return acc(net,Xte,yte)
def conf_learning(D,K,c,seed,folds=3):
    Xtr,yn,Xte,yte,yt=D; n=len(yn); rng=np.random.RandomState(seed); fold=rng.permutation(n)%folds; P=np.zeros((n,K),np.float32)
    for f in range(folds):
        tr=fold!=f; net=train_ce(Xtr[tr],yn[tr],K,c,seed+f); P[~tr]=net.predict(Xtr[~tr])[0]
    th=np.array([P[yn==j,j].mean() if (yn==j).any() else 1.0 for j in range(K)])
    above=P>=th[None,:]; Pm=np.where(above,P,-1); guess=Pm.argmax(1); has=above.any(1)
    noisy=has&(guess!=yn); keep=~noisy
    net=train_ce(Xtr[keep],yn[keep],K,c,seed); prec=float((yn!=yt)[noisy].mean()) if noisy.any() else 0.0
    return acc(net,Xte,yte),dict(cl_flag=float(noisy.mean()),cl_prec=prec)
def enn(D,K,c,seed,k=5):
    Xtr,yn,Xte,yte,yt=D; idx=core.knn_graph(Xtr,k); maj=np.array([np.bincount(yn[r],minlength=K).argmax() for r in idx]); keep=maj==yn
    net=train_ce(Xtr[keep],yn[keep],K,c,seed); return acc(net,Xte,yte)
def run_method(m,D,K,c,seed,tau):
    if m in ('ce','gce','sce','apl','smallloss','dynec','dacon'):
        meth='dynec' if m=='dacon' else m; fl={'dep':True} if m=='dacon' else None
        a,aux=core.run(D,meth,c['epochs'],c['warm'],c['bs'],c['sizes'],seed,K,flags=fl,log_auc=meth in('dynec','smallloss')); return a,aux
    if m=='coteach': return coteach(D,K,c,seed,tau),{}
    if m=='jocor': return coteach(D,K,c,seed,tau,joint=True),{}
    if m=='dividemix': return dividemix_lite(D,K,c,seed),{}
    if m=='selc': return selc(D,K,c,seed),{}
    if m=='cl': return conf_learning(D,K,c,seed)
    if m=='enn': return enn(D,K,c,seed),{}
