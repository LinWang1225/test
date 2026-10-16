import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from kvbench_asym.inputs import (LONG_TASKS,inventory,source_bytes,parse_records,prepare,fit,iter_source)
from kvbench_asym.common import atomic_json,sha_file,jsonl


class Tokenizer:
    def encode(self,prompt,**kwargs):return list(prompt.encode())
    def decode(self,ids,**kwargs):return bytes(ids).decode(errors='ignore')
    def apply_chat_template(self,messages,**kwargs):return [1,2]+self.encode(messages[0]['content'])+[3]


def make_zip(path,missing=None):
    with zipfile.ZipFile(path,'w') as z:
        for name in LONG_TASKS:
            if name==missing:continue
            rows=[dict(_id=str(i),context='context',input='question',answers=['answer']) for i in range(2)]
            z.writestr('data/'+name+'.jsonl','\n'.join(json.dumps(r) for r in rows))


class InputTests(unittest.TestCase):
    def test_all_21_tasks_and_all_rows(self):
        with tempfile.TemporaryDirectory() as d:
            z=Path(d)/'data.zip';make_zip(z)
            inv=inventory({'settings':{'source':str(z)}},['longbench_all'])
            self.assertEqual(len(inv),21);self.assertEqual(sum(r['count'] for r in inv),42)
            m=prepare(Path(d)/'inputs',inv,None,4000,96,True,'head-tail',tokenizer=Tokenizer())
            self.assertEqual(m['cases'],42);self.assertTrue(m['all_source_records_selected'])
    def test_missing_task_does_not_fallback_to_six(self):
        with tempfile.TemporaryDirectory() as d:
            z=Path(d)/'data.zip';make_zip(z,'lcc')
            with self.assertRaises(ValueError):inventory({'settings':{'source':str(z)}},['longbench_all'])
    def test_duplicate_member_refused(self):
        with tempfile.TemporaryDirectory() as d:
            z=Path(d)/'data.zip'
            with zipfile.ZipFile(z,'w') as f:
                f.writestr('qasper.jsonl','{}');f.writestr('data/qasper.jsonl','{}')
            with self.assertRaises(ValueError):source_bytes(z,'qasper')
    def test_zip_not_extracted(self):
        with tempfile.TemporaryDirectory() as d:
            z=Path(d)/'data.zip'
            with zipfile.ZipFile(z,'w') as f:f.writestr('../owned','bad');f.writestr('qasper.jsonl','{}')
            source_bytes(z,'qasper');self.assertFalse((Path(d)/'owned').exists())
    def test_duplicate_dataset_scope_refused(self):
        with self.assertRaises(ValueError):inventory({'settings':{}},['longbench_all','longbench_frozen'])
    def test_clipping_preserves_count(self):
        p,orig,n,clipped=fit(Tokenizer(),'x'*1000,'chat',True,90,'head-tail')
        self.assertTrue(clipped);self.assertEqual(orig,1003);self.assertLessEqual(n,90)
    def test_overflow_error_explicit(self):
        with self.assertRaises(ValueError):fit(Tokenizer(),'x'*1000,'chat',True,90,'error')
    def test_no_clipping_for_short_prompt(self):
        self.assertEqual(fit(Tokenizer(),'abc','completion',False,100,'error'),('abc',3,3,False))
    def test_pilot_limit_explicit(self):
        with tempfile.TemporaryDirectory() as d:
            z=Path(d)/'data.zip';make_zip(z)
            inv=inventory({'settings':{'source':str(z)}},['longbench_all'])
            m=prepare(Path(d)/'inputs',inv,None,4000,96,True,'head-tail',3,Tokenizer())
            self.assertEqual(m['cases'],3);self.assertFalse(m['all_source_records_selected'])
    def test_real_data_hash_and_full_count(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'humaneval.jsonl'
            p.write_text('\n'.join(json.dumps(dict(case_id=str(i),messages=[dict(role='user',content='safe prompt')],reference='SECRET')) for i in range(164)))
            atomic_json(root/'manifest.json',{'datasets':{'humaneval':{'file':p.name,'sha256':sha_file(p),'count':164}}})
            inv=inventory({'settings':{}},['humaneval'],real_data=root)
            self.assertEqual(inv[0]['count'],164)
            rows=list(iter_source(inv[0]));self.assertTrue(all('SECRET' not in r['prompt'] for r in rows))
    def test_wrong_math_count_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'math_500.jsonl';p.write_text('{}\n')
            atomic_json(root/'manifest.json',{'datasets':{'math_500':{'file':p.name,'sha256':sha_file(p),'count':1}}})
            with self.assertRaises(ValueError):inventory({'settings':{}},['math_500'],real_data=root)
    def test_invalid_record(self):
        with self.assertRaises(ValueError):parse_records(b'[]','test')
    def test_template_coverage(self):
        p=Path(__file__).parents[1]/'longbench_prompts.json'
        self.assertEqual(set(json.loads(p.read_text())),set(LONG_TASKS))
