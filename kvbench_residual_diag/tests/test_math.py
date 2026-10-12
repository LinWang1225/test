from __future__ import annotations
import itertools
import unittest
import numpy as np
from kvbench_residual_diag.math_ops import (Group,evaluate,objective,action_setup,select_order,
    apply_budget,v_lower_bound,exact_small,mass_upper,bound_policy,lse)
from kvbench_residual_diag.common import output_metrics


def make_group(seed=1,n=4,h=3,d=5,pool=True):
    rng=np.random.default_rng(seed)
    raw=[];chunks=[]
    for i in range(n):
        sa=rng.normal(size=(h,4))
        sf=sa+rng.normal(0,.12,size=(h,4))
        va=rng.normal(size=(4,d))
        vf=va+rng.normal(0,.08,size=(4,d))
        quant=not(pool and i==0)
        if not quant: sa=sf.copy();va=vf.copy()
        raw.append((sa,sf,va,vf))
        B=float(np.linalg.norm(vf,axis=1).max())+1
        eta=float(np.linalg.norm(vf-va,axis=1).max())+.000001 if quant else 0
        chunks.append({"quantized":quant,"k_bytes":32 if quant else 0,"v_bytes":32 if quant else 0,
            "native_code_bytes":128 if quant else 0,"tokens":4,"block":i,"chunk":0,"kv_head":0,
            "seq_blocks":n,"fraction":1/n,"lse_a":lse(sa),"lse_f":lse(sf),
            "epsilon_k":np.max(abs(sa-sf),axis=1)+1e-7,"actual_k_logit_error":np.max(abs(sa-sf),axis=1),
            "v_bound":B,"eta_v":eta,"v_ratio":eta/B})
    origin=np.stack([np.maximum(x[0].max(axis=1),x[1].max(axis=1)) for x in raw]).max(axis=0)
    z=np.empty((n,2,h));u=np.empty((n,2,2,h,d))
    for i,(sa,sf,va,vf) in enumerate(raw):
        for k,s in enumerate((sa,sf)):
            e=np.exp(s-origin[:,None]);z[i,k]=e.sum(axis=1)
            u[i,k,0]=e@va;u[i,k,1]=e@vf
    ref=u[:,1,1].sum(axis=0)/z[:,1].sum(axis=0)[:,None]
    return Group(z,u,chunks,max(c['v_bound'] for c in chunks),ref,origin),raw


class MathTests(unittest.TestCase):
    def test_sufficient_statistics_match_dense(self):
        g,raw=make_group()
        for kb in itertools.product((0,1),repeat=g.n):
            vb=tuple(reversed(kb))
            score=np.concatenate([raw[i][kb[i]] for i in range(g.n)],axis=1)
            values=np.concatenate([raw[i][2+vb[i]] for i in range(g.n)],axis=0)
            probs=np.exp(score-score.max(axis=1,keepdims=True));probs/=probs.sum(axis=1,keepdims=True)
            np.testing.assert_allclose(evaluate(g,kb,vb),probs@values,rtol=1e-13,atol=1e-13)
    def test_full_read_exact(self):
        g,_=make_group()
        np.testing.assert_array_equal(evaluate(g,np.ones(g.n),np.ones(g.n)),g.full)
    def test_v_linear_decomposition(self):
        g,_=make_group()
        k,v,_=action_setup(g,'v')
        base=evaluate(g,k,v)
        delta=(g.u[:,1,1]-g.u[:,1,0])/g.z[:,1].sum(axis=0)[None,:,None]
        np.testing.assert_allclose(g.full-base,delta.sum(axis=0),atol=1e-14)
    def test_gqa_mask_shape_rejected(self):
        g,_=make_group()
        with self.assertRaises(ValueError): evaluate(g,np.ones((g.n,3)),np.ones(g.n))
    def test_zero_output_relative_is_na(self):
        x=output_metrics(np.ones((3,4)),np.zeros((3,4)))
        self.assertIsNone(x['relative_l2_max_head']);self.assertIsNone(x['relative_l2_fro'])
        self.assertEqual(x['relative_undefined_heads'],3)
    def test_lower_bound_valid_against_exact(self):
        for seed in range(7):
            g,_=make_group(seed,n=5)
            for budget in range(0,129,16):
                ex=exact_small(g,'v',budget,8)
                self.assertLessEqual(v_lower_bound(g,budget),ex[0]+1e-12)
    def test_lower_bound_zero_for_full_read(self):
        g,_=make_group()
        self.assertEqual(v_lower_bound(g,int(g.costs[:,1].sum())),0)
    def test_exact_solver_refuses_large(self):
        g,_=make_group(n=6)
        self.assertIsNone(exact_small(g,'kv',64,max_actions=4))
    def test_budget_never_overspent(self):
        g,_=make_group(n=6)
        for mode in ('k','v','kv'):
            for kind in ('attention','random','greedy'):
                order=select_order(g,mode,kind)
                for budget in range(0,330,21):
                    k,v,spent=apply_budget(g,mode,order,budget)
                    self.assertLessEqual(spent,budget)
                    _,_,actions=action_setup(g,mode)
                    actual=sum(g.costs[i,s]*(k[i] if s==0 else v[i]) for i,s in actions)
                    self.assertEqual(spent,actual)
    def test_best_prefix_monotone(self):
        g,_=make_group(seed=14,n=6)
        for mode in ('k','v','kv'):
            order=select_order(g,mode,'greedy');last=float('inf')
            for budget in range(0,321,16):
                k,v,_=apply_budget(g,mode,order,budget)
                now=objective(evaluate(g,k,v),g.full)
                self.assertLessEqual(now,last+1e-13);last=now
    def test_full_budget_reaches_full_fidelity(self):
        g,_=make_group(n=5)
        for mode in ('k','v','kv'):
            for kind in ('attention','greedy','random'):
                order=select_order(g,mode,kind)
                k,v,_=apply_budget(g,mode,order,10000)
                self.assertLess(objective(evaluate(g,k,v),g.full),1e-13)
    def test_exact_no_worse_than_greedy(self):
        g,_=make_group(n=5)
        for mode in ('k','v','kv'):
            order=select_order(g,mode,'greedy')
            for budget in (0,32,64,128):
                k,v,_=apply_budget(g,mode,order,budget)
                self.assertLessEqual(exact_small(g,mode,budget,12)[0],objective(evaluate(g,k,v),g.full)+1e-13)
    def test_order_unique_and_reproducible(self):
        g,_=make_group(n=6)
        for mode in ('k','v','kv'):
            for kind in ('attention','greedy','random'):
                x=select_order(g,mode,kind,34);y=select_order(g,mode,kind,34)
                self.assertEqual(x,y);self.assertEqual(len(x),len(set(x)))
                self.assertEqual(set(x),set(action_setup(g,mode)[2]))
    def test_pool_always_full(self):
        g,_=make_group()
        for mode in ('k','v','kv'):
            k,v,_=action_setup(g,mode)
            self.assertTrue(k[0]);self.assertTrue(v[0])
    def test_mass_prefix_empty_is_one(self):
        m,_=mass_upper(np.array([3.,-1.]),np.array([.1,.2]),np.array([-np.inf,-np.inf]))
        np.testing.assert_allclose(m,[1.,1.])
    def test_larger_prefix_tightens_mass(self):
        a,_=mass_upper(np.array([2.]),np.array([.1]),np.array([-5.]))
        b,_=mass_upper(np.array([2.]),np.array([.1]),np.array([3.]))
        self.assertLessEqual(b[0],a[0])
    def test_bound_zero_reads_full(self):
        g,_=make_group()
        for mode in ('bound_k','bound_v','bound_kv'):
            b=bound_policy(g,mode,0,1)
            self.assertTrue(b['read_k'].all());self.assertTrue(b['read_v'].all())
    def test_bound_budget_real_arithmetic(self):
        g,_=make_group(n=5)
        for p in ('bound_k','bound_v','bound_kv'):
            for e in (.01,.1,1.,10.):
                for sp in (1,2,4):
                    b=bound_policy(g,p,e,sp)
                    error=np.linalg.norm(b['output']-g.full,axis=1).max()
                    self.assertLessEqual(error,e*g.vref+1e-12)
    def test_anchor_and_v_only_code_limits(self):
        g,_=make_group()
        a=bound_policy(g,'anchor',.02,1)
        self.assertAlmostEqual(a['logical_code_reduction_fraction'],.5)
        k,v,_=action_setup(g,'v')
        costs=g.costs
        full=int(costs.sum())
        read=int((costs[:,0]*k+costs[:,1]*v).sum())
        self.assertAlmostEqual(1-(full+read)/(2*full),.25)
    def test_nonfinite_output_rejected(self):
        with self.assertRaises(ValueError): output_metrics(np.array([[np.nan]]),np.array([[1.]]))

if __name__=='__main__': unittest.main()
