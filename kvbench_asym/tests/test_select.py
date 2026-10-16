import unittest
from dataclasses import replace
from decimal import Decimal
import numpy as np
from kvbench_asym.select import from_group,quota,choose,accounting
from kvbench_asym.analyze import group_rows
from kvbench_asym.tests.helpers import group


class SelectionTests(unittest.TestCase):
    def test_equal_integer_total_for_all_shares(self):
        for n in (0,1,2,3,9,127,511):
            for t in ('0','.3','1','1.1','1.9','2'):
                values=[quota(n,t,s) for s in ('0','.25','.5','.75','1')]
                self.assertEqual(len({x[2] for x in values}),1)
                for k,v,m in values:
                    self.assertEqual(k+v,m);self.assertTrue(0<=k<=n and 0<=v<=n)
    def test_exact_decimal_floor(self):
        self.assertEqual(quota(10,'1.1','.5'),(6,5,11))
    def test_endpoints(self):
        self.assertEqual(quota(100,'1.1','1'),(100,10,110))
        self.assertEqual(quota(100,'1.1','0'),(10,100,110))
        self.assertEqual(quota(100,'1.1','.5'),(55,55,110))
    def test_invalid_quotas(self):
        for t,s,n in [('NaN',.5,2),(3,.5,3),(1,-1,3),(1,.5,-1),('Infinity',.5,2)]:
            with self.assertRaises(ValueError):quota(n,t,s)
    def test_exact_counts_all_selectors(self):
        g=group();v=from_group(g,16);n=v.eligible.sum()
        for method in ('anchor_mass','staged_mass','random'):
            for nk in range(n+1):
                for nv in range(n+1):
                    k,val=choose(v,nk,nv,method)
                    self.assertEqual(k[v.eligible].sum(),nk);self.assertEqual(val[v.eligible].sum(),nv)
    def test_protected_pool(self):
        v=from_group(group(),16)
        for m in ('anchor_mass','staged_mass','random'):
            k,val=choose(v,0,0,m);self.assertTrue(k[0] and val[0])
    def test_partial_chunk_full_read(self):
        v=from_group(group(partial=True),16)
        self.assertFalse(v.eligible[-1])
        k,val=choose(v,0,0,'staged_mass');self.assertTrue(k[-1] and val[-1])
    def test_byte_equality_with_partial_boundary(self):
        g=group(partial=True)
        rows,_,_=group_rows(g,16,['1.1'],['0','.5','1'],['staged_mass','random'],[42],dict(dataset='x'))
        values=[r['logical_payload_read_bytes'] for r in rows if r['kind']=='paired']
        self.assertEqual(len(set(values)),1)
    def test_no_quant_history_ineligible(self):
        g=group(no_quant=True)
        rows,_,_=group_rows(g,16,['1.1'],['0','.5','1'],['staged_mass'],[42],{})
        self.assertTrue(all(not r['eligible_for_equal_payload'] for r in rows))
        self.assertTrue(all(r['payload_reduction_fraction'] is None for r in rows))
    def test_no_unread_key_leak_into_value_ranking(self):
        v=from_group(group(),16)
        k,mask=choose(v,3,2,'staged_mass')
        changed=v.lse_full_key.copy();changed[~k]=1e8
        k2,mask2=choose(replace(v,lse_full_key=changed),3,2,'staged_mass')
        np.testing.assert_array_equal(k,k2);np.testing.assert_array_equal(mask,mask2)
    def test_anchor_ranking_never_uses_full_key(self):
        v=from_group(group(),16)
        x=choose(v,3,4,'anchor_mass')
        y=choose(replace(v,lse_full_key=np.full_like(v.lse_full_key,np.nan)),3,4,'anchor_mass')
        for a,b in zip(x,y):np.testing.assert_array_equal(a,b)
    def test_masks_do_not_depend_on_reference_output(self):
        g=group();a=from_group(g,16)
        g.full[:]=1e9
        b=from_group(g,16)
        for method in ('staged_mass','anchor_mass','random'):
            for x,y in zip(choose(a,2,3,method),choose(b,2,3,method)):np.testing.assert_array_equal(x,y)
    def test_random_reproducible(self):
        v=from_group(group(n=30),16)
        for a,b in zip(choose(v,10,10,'random',42),choose(v,10,10,'random',42)):np.testing.assert_array_equal(a,b)
    def test_random_seeds_distinct(self):
        v=from_group(group(n=30),16)
        self.assertFalse(np.array_equal(choose(v,10,10,'random',42)[0],choose(v,10,10,'random',43)[0]))
    def test_full_control_zero_error(self):
        rows,_,_=group_rows(group(),16,['1'],['.5'],['staged_mass'],[42],{})
        self.assertEqual(rows[0]['relative_l2_max_head'],0.)
    def test_masks_pack_roundtrip(self):
        rows,packed,n=group_rows(group(),16,['1.1'],['0','.5','1'],['staged_mass'],[42],{})
        masks=np.unpackbits(packed,axis=-1)[...,:n]
        self.assertEqual(masks.shape,(len(rows),2,n));self.assertTrue(masks[0].all())
    def test_v_only_code_saving(self):
        v=from_group(group(n=11),16)
        k,rv=choose(v,10,1,'staged_mass')
        self.assertAlmostEqual(accounting(v,k,rv)['payload_reduction_fraction'],.225)
    def test_fixed_budget_spent_even_when_error_worsens(self):
        g=group(seed=6)
        rows,_,_=group_rows(g,16,['1.1'],['0','.5','1'],['random'],[42],{})
        v=from_group(g,16);m=quota(v.eligible.sum(),'1.1','.5')[2]
        for r in rows[2:]:
            self.assertEqual(r['key_residual_bytes']+r['value_residual_bytes'],m*v.packet_bytes)
    def test_invalid_method(self):
        with self.assertRaises(ValueError):choose(from_group(group(),16),1,1,'oracle')
    def test_quota_overspend_rejected(self):
        with self.assertRaises(ValueError):choose(from_group(group(),16),50,1,'random')
