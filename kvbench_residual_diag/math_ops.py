"""Float64 counterfactual attention on the SAME captured Q/KV.

Full quantized attention is the reference, not an FP16-model output. Residual
selection is shared by all query heads in a GQA group. A block is never deleted:
its anchor numerator and denominator always participate in attention.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np


@dataclass
class Group:
    z: np.ndarray              # [chunks, key_state=2, query_heads]
    u: np.ndarray              # [chunks, key_state=2, value_state=2, query_heads, dim]
    chunks: list
    vref: float
    full: np.ndarray
    origin: np.ndarray

    @property
    def n(self): return len(self.chunks)
    @property
    def quantized(self): return np.array([c["quantized"] for c in self.chunks], bool)
    @property
    def costs(self): return np.array([[c["k_bytes"],c["v_bytes"]] for c in self.chunks],np.int64)


def lse(a):
    a = np.asarray(a, np.float64)
    m = a.max(axis=-1)
    return m + np.log(np.exp(a-m[...,None]).sum(axis=-1))


def mass_upper(log_mass, epsilon, prefix):
    lower = np.logaddexp(prefix, log_mass-epsilon)
    return np.exp(np.minimum(0.0,log_mass+epsilon-lower)),lower


def build_group(data, hk, block_n=32):
    from kvbench_residual.fixtures import unpack_block
    c, cfg = data["case"], data["case"].layout
    if len(c.seq_lens) != 1 or block_n not in (16,32,64):
        raise ValueError("Group builder requires C1 and BLOCK_N in {16,32,64}")
    L,g,d = c.seq_lens[0],cfg.group,cfg.head_dim
    ratio = c.hq//c.hk
    if hk < 0 or hk >= c.hk:
        raise ValueError("Invalid KV head")
    q = data["query"][0,hk*ratio:(hk+1)*ratio].astype(np.float64)
    first = max(0,L-c.window) if c.window else 0
    raw,chunks = [],[]
    vref = 0.0
    for block in range(math.ceil(L/g)):
        bid = int(data["block_table"][0,block])
        slot = int(data["block_to_slot"][bid])
        unpacked = unpack_block(data,bid,hk) if slot < 0 else None
        for begin in range(0,g,block_n):
            cols = np.arange(begin,begin+block_n)
            cols = cols[(block*g+cols >= first)&(block*g+cols < L)]
            if not len(cols): continue
            if slot >= 0:
                Kf = data["pool_k"][slot,cols,hk].astype(np.float64).T
                Vf = data["pool_v"][slot,cols,hk].astype(np.float64)
                Ka,Va = Kf,Vf
                eb = np.zeros(ratio)
                vrange = eta = B = 0.0
                vref = max(vref,float(np.linalg.norm(Vf,axis=1).max()))
            else:
                x = unpacked
                kq,vq = x["kq"][:,cols],x["vq"][cols]
                sk,zk,rk = x["sk"],x["zk"],x["rk"][cols]
                sv,rv,zv = x["sv"],x["rv"][cols],x["zv"][cols]
                Kf = (kq*sk[:,None]+zk[:,None])*rk[None,:]
                Ka = ((4*np.floor(kq/4)+1.5)*sk[:,None]+zk[:,None])*rk[None,:]
                Vf = (vq*rv[:,None]+zv[:,None])*sv[None,:]
                Va = ((4*np.floor(vq/4)+1.5)*rv[:,None]+zv[:,None])*sv[None,:]
                eb = 1.5*np.max(np.abs(rk))*np.sum(np.abs(q)*np.abs(sk)[None,:],axis=1)*abs(data["scale"])
                eb = eb*1.000001+1e-7
                vrange = float(np.maximum(np.abs(zv),np.abs(zv+15*rv)).max())
                eta = 1.5*float(np.max(np.abs(rv)))*float(np.linalg.norm(sv))
                B = vrange*float(np.linalg.norm(sv))
                vref = max(vref,B)
            sf,sa = q@Kf*data["scale"],q@Ka*data["scale"]
            if not all(np.isfinite(t).all() for t in (sf,sa,Vf,Va,eb)):
                raise ValueError("Nonfinite decoded active cache")
            kb = (int(cols[-1])//4-int(cols[0])//4+1)*d if slot < 0 else 0
            vb = len(cols)*d//4 if slot < 0 else 0
            chunks.append({"block":block,"chunk":begin,"tokens":len(cols),"quantized":slot<0,
                           "kv_head":hk,"k_bytes":kb,"v_bytes":vb,"native_code_bytes":block_n*d if slot<0 else 0,
                           "lse_a":lse(sa),"lse_f":lse(sf),"epsilon_k":eb,
                           "actual_k_logit_error":np.max(np.abs(sf-sa),axis=1),
                           "v_bound":B,"eta_v":eta,"v_ratio":eta/B if B>0 else 0.0,
                           "fraction":len(cols)/(L-first),"seq_blocks":math.ceil(L/g)})
            raw.append((sa,sf,Va,Vf))
    if not raw: raise ValueError("No attended chunks")
    origin = np.stack([np.maximum(sa.max(axis=1),sf.max(axis=1)) for sa,sf,_,_ in raw]).max(axis=0)
    z = np.empty((len(raw),2,ratio),np.float64)
    u = np.empty((len(raw),2,2,ratio,d),np.float64)
    for i,(sa,sf,Va,Vf) in enumerate(raw):
        for ki,s in enumerate((sa,sf)):
            p = np.exp(s-origin[:,None])
            z[i,ki] = p.sum(axis=1)
            u[i,ki,0] = p@Va
            u[i,ki,1] = p@Vf
    full = u[:,1,1].sum(axis=0)/z[:,1].sum(axis=0)[:,None]
    if not np.isfinite(full).all(): raise ValueError("Invalid complete-quantized output")
    return Group(z,u,chunks,vref,full,origin)


def evaluate(group, read_k, read_v):
    k,v = np.asarray(read_k,np.int64),np.asarray(read_v,np.int64)
    if k.shape != (group.n,) or v.shape != k.shape or np.any((k<0)|(k>1)|(v<0)|(v>1)):
        raise ValueError("Invalid shared GQA masks")
    i = np.arange(group.n)
    z = group.z[i,k].sum(axis=0)
    u = group.u[i,k,v].sum(axis=0)
    if np.any(z <= 0): raise ValueError("Counterfactual denominator underflow")
    return u/z[:,None]


def bound_policy(group, policy, epsilon, splits):
    """Reproduce v1's real-arithmetic decisions, plus reasons for each rejection.
    Diagnostic actual mass is explanatory only, never used by the controller.
    """
    if policy not in ("parent","repacked_full","anchor","bound_k","bound_v","bound_kv"):
        raise ValueError(policy)
    if not np.isfinite(epsilon) or epsilon < 0 or splits<1 or splits&(splits-1):
        raise ValueError("Invalid epsilon/splits")
    h = group.full.shape[0]
    read_k,read_v = np.ones(group.n,bool),np.ones(group.n,bool)
    tau_k = epsilon if policy=="bound_k" else epsilon/2 if policy=="bound_kv" else 0.0
    tau_v = epsilon if policy=="bound_v" else epsilon/2 if policy=="bound_kv" else 0.0
    blocks_per = math.ceil(group.chunks[0]["seq_blocks"]/splits)
    previous = None
    prefix = np.full(h,-np.inf)
    details = []
    full_denom = group.z[:,1].sum(axis=0)
    for i,c in enumerate(group.chunks):
        split = c["block"]//blocks_per
        if split != previous:
            prefix = np.full(h,-np.inf)
            previous = split
        if not c["quantized"]:
            if policy.startswith("bound_"):
                _,prefix = mass_upper(c["lse_f"],np.zeros(h),prefix)
            continue
        kp = np.zeros(h,bool)
        vp = np.zeros(h,bool)
        k_term = np.zeros(h)
        k_upper = None
        k_ratio = v_ratio = None
        err = np.zeros(h)
        used_lse = c["lse_f"]
        if policy in ("anchor","bound_k","bound_kv"):
            k_upper,_ = mass_upper(c["lse_a"],c["epsilon_k"],prefix)
            k_term = 2*k_upper*c["epsilon_k"]
            kp = k_term <= tau_k*c["fraction"]
            if tau_k > 0:
                k_ratio = k_term/(tau_k*c["fraction"])
            read_k[i] = not (policy=="anchor" or kp.all())
            if not read_k[i]:
                err = c["epsilon_k"]
                used_lse = c["lse_a"]
        if policy.startswith("bound_"):
            v_upper,prefix = mass_upper(used_lse,err,prefix)
        else:
            v_upper = None
        if policy=="anchor": read_v[i] = False
        elif policy in ("bound_v","bound_kv"):
            v_term = v_upper*c["v_ratio"]*1.000001
            vp = v_term <= tau_v*c["fraction"]
            if tau_v > 0: v_ratio = v_term/(tau_v*c["fraction"])
            read_v[i] = not vp.all()
        true_mass = group.z[i,1]/full_denom
        actual = c["actual_k_logit_error"]
        positive = actual>1e-15
        details.append({"kv_head":c["kv_head"],"split":split,"block":c["block"],"chunk":c["chunk"],
            "tokens":c["tokens"],"skip_k_cpu":not bool(read_k[i]),"skip_v_cpu":not bool(read_v[i]),
            "k_individual_pass_heads":int(kp.sum()) if tau_k>0 else None,
            "v_individual_pass_heads":int(vp.sum()) if tau_v>0 else None,
            "gqa_heads":h,"k_gqa_prevented_skip":bool(kp.any() and not kp.all()) if tau_k>0 else None,
            "v_gqa_prevented_skip":bool(vp.any() and not vp.all()) if tau_v>0 else None,
            "k_threshold_ratio_max":float(k_ratio.max()) if k_ratio is not None else None,
            "k_threshold_ratio_min":float(k_ratio.min()) if k_ratio is not None else None,
            "v_threshold_ratio_max":float(v_ratio.max()) if v_ratio is not None else None,
            "v_threshold_ratio_min":float(v_ratio.min()) if v_ratio is not None else None,
            "k_bound_max":float(c["epsilon_k"].max()),"k_actual_logit_error_max":float(actual.max()),
            "k_bound_over_actual_max":float((c["epsilon_k"][positive]/actual[positive]).max()) if positive.any() else None,
            "full_attention_mass_max":float(true_mass.max()),
            "k_mass_upper_max":float(k_upper.max()) if k_upper is not None else None,
            "v_mass_upper_max":float(v_upper.max()) if v_upper is not None else None,
            "k_bytes":c["k_bytes"],"v_bytes":c["v_bytes"]})
    quant = group.quantized
    cost = group.costs
    residual = int((cost[:,0]*read_k+cost[:,1]*read_v).sum())
    anchor = int(cost.sum())
    native = int(sum(c["native_code_bytes"] for c in group.chunks))
    return {"output":evaluate(group,read_k,read_v),"read_k":read_k,"read_v":read_v,
            "details":details,"key_skipped_chunks":int((quant&~read_k).sum()),
            "value_skipped_chunks":int((quant&~read_v).sum()),"quantized_chunks":int(quant.sum()),
            "logical_code_bytes":anchor+residual,"native_code_bytes":native,
            "logical_code_reduction_fraction":1-(anchor+residual)/native if native else None}


def action_setup(group, mode):
    if mode not in ("k","v","kv"): raise ValueError(mode)
    k,v = np.ones(group.n,bool),np.ones(group.n,bool)
    if "k" in mode: k[group.quantized] = False
    if "v" in mode: v[group.quantized] = False
    actions = [(i,s) for i,c in enumerate(group.chunks) if c["quantized"]
               for s in (0,1) if (s==0 and "k" in mode) or (s==1 and "v" in mode)]
    return k,v,actions


def objective(output, reference):
    # Used only for ranking. Reports retain undefined near-zero-relative heads.
    den = np.maximum(np.linalg.norm(reference,axis=1),1e-12)
    return float(np.max(np.linalg.norm(output-reference,axis=1)/den))


def select_order(group,mode,kind,seed=0):
    """Return a feasible, offline ordering. Greedy is NOT a certified optimum."""
    k,v,actions = action_setup(group,mode)
    if not actions: return []
    if kind=="random":
        idx = np.random.default_rng(seed).permutation(len(actions))
        return [actions[j] for j in idx]
    if kind=="attention":
        # Implementable proxy uses anchor logits for K/KV; V-only already reads K fully.
        key_state = 1 if mode=="v" else 0
        masses = group.z[:,key_state]/group.z[:,key_state].sum(axis=0)
        rank = [float(masses[i].max())/max(1,group.costs[i,s]) for i,s in actions]
        return [actions[j] for j in np.argsort(-np.array(rank),kind="stable")]
    if kind!="greedy": raise ValueError(kind)
    # Vectorized exact one-action lookahead on the full recorded residuals.
    # Negative immediate improvements are allowed; cancellation can make the path nonmonotone.
    order=[]
    remaining=list(actions)
    indices=np.arange(group.n)
    while remaining:
        z=group.z[indices,k.astype(int)].sum(axis=0)
        u=group.u[indices,k.astype(int),v.astype(int)].sum(axis=0)
        ii=np.array([a[0] for a in remaining],int)
        side=np.array([a[1] for a in remaining],int)
        kk=k[ii].astype(int); vv=v[ii].astype(int)
        kn=np.where(side==0,1,kk); vn=np.where(side==1,1,vv)
        dz=group.z[ii,kn]-group.z[ii,kk]
        du=group.u[ii,kn,vn]-group.u[ii,kk,vv]
        candidates=(u[None,:,:]+du)/(z[None,:]+dz)[:,:,None]
        den=np.maximum(np.linalg.norm(group.full,axis=1),1e-12)
        values=np.max(np.linalg.norm(candidates-group.full[None,:,:],axis=2)/den[None,:],axis=1)
        now=objective(u/z[:,None],group.full)
        costs=group.costs[ii,side]
        gain=(now-values)/costs
        j=int(np.argmax(gain))
        i,s=remaining.pop(j)
        if s==0: k[i]=True
        else: v[i]=True
        order.append((i,s))
    return order


def apply_budget(group,mode,order,budget):
    """A nested prefix only. Never overspend a byte budget; actual spend is reported."""
    k,v,_=action_setup(group,mode)
    spent=0
    best_k,best_v=k.copy(),v.copy()
    best=objective(evaluate(group,k,v),group.full)
    best_spent=0
    for i,s in order:
        cost=int(group.costs[i,s])
        if spent+cost>budget: break
        spent+=cost
        if s==0: k[i]=True
        else: v[i]=True
        value=objective(evaluate(group,k,v),group.full)
        # Keep the best feasible prefix, because cancellation defeats monotonicity.
        if value < best or (value==best and spent<best_spent):
            best=value;best_k=k.copy();best_v=v.copy();best_spent=spent
    return best_k,best_v,best_spent


def v_lower_bound(group,budget):
    """Rigorous triangle/fractional-knapsack lower bound on optimum V-only error.
    This is often zero and is then inconclusive, NOT a prediction of zero error.
    K is fixed full; all query heads share the same mask. Relaxing that mask per
    head only weakens the bound, so max over per-head lower bounds remains valid.
    """
    ix=np.flatnonzero(group.quantized)
    if not len(ix): return 0.0
    denom=group.z[:,1].sum(axis=0)
    delta=(group.u[ix,1,1]-group.u[ix,1,0])/denom[None,:,None]
    total=np.linalg.norm(delta.sum(axis=0),axis=1)
    norms=np.linalg.norm(delta,axis=2)
    costs=group.costs[ix,1]
    ref=np.linalg.norm(group.full,axis=1)
    if np.any(ref<=1e-12): return None
    lower=[]
    for h in range(len(ref)):
        order=np.argsort(-norms[:,h]/costs,kind="stable")
        left=float(budget); removable=0.0
        for j in order:
            take=min(left,float(costs[j]))
            removable+=norms[j,h]*take/costs[j]
            left-=take
            if left<=0: break
        lower.append(max(0.0,total[h]-removable)/ref[h])
    return float(max(lower))


def exact_small(group,mode,budget,max_actions=14):
    """Exhaustive optimum for THIS small captured group/byte budget/objective.
    Larger problems return None; no approximate solution is called exact.
    """
    k0,v0,actions=action_setup(group,mode)
    n=len(actions)
    if n>max_actions: return None
    costs=np.array([group.costs[i,s] for i,s in actions])
    best=(float("inf"),None,None,None)
    for mask in range(1<<n):
        take=np.array([(mask>>j)&1 for j in range(n)],bool)
        spent=int(costs[take].sum())
        if spent>budget: continue
        k,v=k0.copy(),v0.copy()
        for use,(i,s) in zip(take,actions):
            if use:
                if s==0: k[i]=True
                else: v[i]=True
        val=objective(evaluate(group,k,v),group.full)
        if val<best[0] or (val==best[0] and spent<best[3]): best=(val,k,v,spent)
    return best
