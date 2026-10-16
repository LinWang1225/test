import csv
import io
import unittest
import numpy as np
from kvbench_asym.report import case_summary,controller_summary,paired_bootstrap
from kvbench_asym.analyze import group_rows
from kvbench_asym.tests.helpers import group


def rows(no_quant=False):
    r,_,_=group_rows(group(no_quant=no_quant),16,['1.1'],['0','.5','1'],['staged_mass'],[42],
                    dict(dataset='test',trace='t',kv_head=0))
    s=io.StringIO();w=csv.DictWriter(s,fieldnames=list(r[0]));w.writeheader();w.writerows(r)
    s.seek(0);return list(csv.DictReader(s))


class ReportTests(unittest.TestCase):
    def test_paired_rows(self):
        s=case_summary(rows());self.assertEqual(len(s),3)
        sym=next(x for x in s if x['k_priority']==.5)
        self.assertEqual(sym['paired_mean_error_difference_vs_symmetric'],0.)
    def test_noninformative_not_zero_success(self):
        self.assertEqual(case_summary(rows(True)),[])
    def test_payload_mismatch_stops_report(self):
        r=rows();r[-1]['logical_payload_read_bytes']='1'
        with self.assertRaises(ValueError):case_summary(r)
    def test_bootstrap_reproducible(self):
        self.assertEqual(paired_bootstrap([1,2,3],100,9),paired_bootstrap([1,2,3],100,9))
    def test_one_case_no_fake_confidence(self):
        self.assertEqual(paired_bootstrap([.1],100,9),(None,None))
    def test_constant_difference_interval(self):
        lo,hi=paired_bootstrap([.1]*20,100,9)
        self.assertAlmostEqual(lo,.1);self.assertAlmostEqual(hi,.1)

    def test_controller_summary_reports_underread_and_rho(self):
        r,_,_=group_rows(group(),16,['1.1'],['.5'],['staged_mass'],[42],
                         dict(dataset='test'),controller_tolerances=(.1,))
        summary=controller_summary(r)
        self.assertEqual(len(summary),1)
        self.assertIn('underread_rate',summary[0])
        self.assertIn('mean_rho',summary[0])
