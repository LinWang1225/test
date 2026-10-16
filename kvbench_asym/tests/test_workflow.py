import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from kvbench_asym.common import atomic_json,sha_file,inside
from kvbench_asym.run import parse,case_valid,prune_new_arrays,validate_capture_coverage
from kvbench_asym.capture import RollingCapture
from kvbench_asym.runtime import check_dependencies


class WorkflowTests(unittest.TestCase):
    def test_defaults_full_dataset(self):
        a=parse([]);self.assertEqual(a.datasets,['longbench_all']);self.assertEqual(a.case_limit,0)
        self.assertEqual(a.cases_per_task,0)

    def test_task_subset_requires_per_task_limit(self):
        with self.assertRaises(SystemExit):
            parse(['--tasks','narrativeqa'])
        a=parse(['--tasks','narrativeqa','qasper','--cases-per-task','10'])
        self.assertEqual(a.cases_per_task,10)
    def test_no_symmetric_control_refused(self):
        with self.assertRaises(SystemExit):parse(['--shares','0','1'])
    def test_late_call_without_room_refused(self):
        with self.assertRaises(SystemExit):parse(['--decode-calls','95'])
    def test_resume_requires_directory(self):
        with self.assertRaises(SystemExit):parse(['--resume'])
    def test_duplicate_profile_refused(self):
        with self.assertRaises(SystemExit):parse(['--totals','1','1.0'])
    def test_package_pins_match_supplied_archive(self):
        self.assertEqual(len(check_dependencies()),4)
    def test_missing_capture_is_not_natural_eos(self):
        with self.assertRaises(RuntimeError):validate_capture_coverage([],{'case_id':'x','dataset':'t'},
                {'completion_tokens':96,'finish_reason':'length'},[0],[0,63])
    def test_early_eos_explicit(self):
        cov=validate_capture_coverage([],{'case_id':'x','dataset':'t'},
                {'completion_tokens':1,'finish_reason':'stop'},[0],[0,63])
        self.assertEqual(len(cov),2);self.assertTrue(all(not r['expected_given_output'] for r in cov))
    def test_complete_capture(self):
        r={'request':{'case_id':'x','dataset':'t'},'cache_ordinal':0,'decode_call_index':0}
        cov=validate_capture_coverage([r],{'case_id':'x','dataset':'t'},
                {'completion_tokens':2,'finish_reason':'stop'},[0],[0,63])
        self.assertTrue(cov[0]['captured']);self.assertFalse(cov[1]['captured'])
    def test_duplicate_capture_rejected(self):
        r={'request':{'case_id':'x','dataset':'t'},'cache_ordinal':0,'decode_call_index':0}
        with self.assertRaises(ValueError):validate_capture_coverage([r,r],{'case_id':'x','dataset':'t'},
                {'completion_tokens':2,'finish_reason':'stop'},[0],[0])
    def test_committed_result_hash_checked(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);base=root/'cases/x';a=base/'attempt';a.mkdir(parents=True)
            f=a/'rows';f.write_text('saved')
            atomic_json(base/'done.json',dict(schema='kvbench.asym.case.v1',status='completed',attempt='cases/x/attempt',file_hashes={'rows':sha_file(f)}))
            self.assertTrue(case_valid(root,base));f.write_text('changed')
            with self.assertRaises(ValueError):case_valid(root,base)
    def test_prune_only_new_named_arrays(self):
        with tempfile.TemporaryDirectory() as d:
            a=Path(d);t=a/'traces';t.mkdir();f=t/'x.npz';f.write_bytes(b'owned')
            keep=t/'other.npz';keep.write_bytes(b'not named')
            atomic_json(t/'trace_1.json',dict(arrays_file=f.name,arrays_sha256=sha_file(f)))
            prune_new_arrays(a);self.assertFalse(f.exists());self.assertTrue(keep.exists())
    def test_path_escape_refused(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):inside(d,'../escape')
    def test_rolling_epoch_resets_caps(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);control=root/'trace_control';control.mkdir();source=root/'source.py';source.write_text('# mock')
            atomic_json(control/'capture_config.json',dict(max_bytes=100,layer_ordinals=[0],decode_calls=[0]))
            cap=RollingCapture(control,source)
            t1=root/'cases/a/traces';t1.mkdir(parents=True)
            atomic_json(control/'arm.json',dict(epoch='a',trace_dir=str(t1)))
            cap.arm();cap.saved=99;cap.bytes=99
            t2=root/'cases/b/traces';t2.mkdir(parents=True)
            atomic_json(control/'arm.json',dict(epoch='b',trace_dir=str(t2)))
            cap.arm();self.assertEqual(cap.saved,0);self.assertEqual(cap.bytes,0);self.assertEqual(cap.root,t2)
    def test_capture_no_arm(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);control=root/'trace_control';control.mkdir();source=root/'source.py';source.write_text('# mock')
            atomic_json(control/'capture_config.json',{})
            cap=RollingCapture(control,source);self.assertIsNone(cap.arm())
