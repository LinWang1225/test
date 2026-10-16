import numpy as np
from kvbench_residual_diag.math_ops import Group, lse


def group(seed=1,n=8,partial=False,no_quant=False):
    rng=np.random.default_rng(seed);h,d=4,8
    raw=[];chunks=[]
    for i in range(n):
        tokens=8 if partial and i==n-1 else 16
        sa=rng.normal(size=(h,tokens));sf=sa+rng.normal(0,.25,size=sa.shape)
        va=rng.normal(size=(tokens,d));vf=va+rng.normal(0,.1,size=va.shape)
        quant=not(no_quant or i==0)
        if not quant:sa=sf.copy();va=vf.copy()
        cost=tokens*d//4 if quant else 0
        chunks.append(dict(quantized=quant,k_bytes=cost,v_bytes=cost,tokens=tokens,
                           lse_a=lse(sa),lse_f=lse(sf),block=i,chunk=0))
        raw.append((sa,sf,va,vf))
    origin=np.stack([np.maximum(a.max(1),b.max(1)) for a,b,_,_ in raw]).max(0)
    z=np.empty((n,2,h));u=np.empty((n,2,2,h,d))
    for i,(sa,sf,va,vf) in enumerate(raw):
        for k,s in enumerate((sa,sf)):
            p=np.exp(s-origin[:,None]);z[i,k]=p.sum(1);u[i,k,0]=p@va;u[i,k,1]=p@vf
    full=u[:,1,1].sum(0)/z[:,1].sum(0)[:,None]
    return Group(z,u,chunks,5.,full,origin)
