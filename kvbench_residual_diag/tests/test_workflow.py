from __future__ import annotations
import json
from pathlib import Path
import tempfile
import unittest
from kvbench_residual_diag.common import atomic_json,new_output,trace_paths,sha_file
from kvbench_residual_diag.run import choose_samples


class WorkflowTests(unittest.TestCase):
    def test_existing_directory_refused(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):new_output(d)
    def test_atomic_finite_json(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.json';atomic_json(p,{'ok':True})
            self.assertEqual(json.loads(p.read_text()),{'ok':True})
            with self.assertRaises(ValueError):atomic_json(p,{'x':float('nan')})
            self.assertEqual(json.loads(p.read_text()),{'ok':True})
    def test_trace_empty_is_error(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):trace_paths(d)
    def test_trace_selection_spreads_range(self):
        with tempfile.TemporaryDirectory() as d:
            for i in range(20): (Path(d)/f'trace_{i:02d}.json').write_text('{}')
            chosen=trace_paths(d,3)
            self.assertEqual(chosen[0].name,'trace_00.json')
            self.assertEqual(chosen[-1].name,'trace_19.json')
            self.assertEqual(len(chosen),3)
    def test_prompt_selection_not_mutated(self):
        rows=[{'dataset':'a','case_id':str(i),'input_tokens':1000+i*100,'prompt':str(i),'max_tokens':None} for i in range(10)]
        chosen=choose_samples({'a':rows},['a'],3,1000,96,32768)
        self.assertEqual(len(chosen),3);self.assertTrue(all(r['max_tokens'] is None for r in rows))
        self.assertEqual(len(set(r['case_id'] for r in chosen)),3)
    def test_insufficient_data_refused(self):
        with self.assertRaises(ValueError): choose_samples({'a':[]},['a'],2,1024,96,32768)
    def test_unknown_task_refused(self):
        with self.assertRaises(ValueError): choose_samples({},['not_a_task'],1,1024,96,32768)
    def test_hash_changes_with_content(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x';p.write_text('abc');s=sha_file(p);p.write_text('xyz')
            self.assertNotEqual(s,sha_file(p))

if __name__=='__main__':unittest.main()
