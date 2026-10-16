"""Mock lifecycle tests only. They do NOT run vLLM, CUDA, or real activations."""
import csv
import gzip
from pathlib import Path
import tempfile
from types import ModuleType,SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np
from kvbench_asym.common import atomic_json,read_json,sha_file
from kvbench_asym.run import main
from kvbench_asym.tests.helpers import group
from kvbench_asym.tests.test_inputs import make_zip,Tokenizer
from kvbench_asym.inputs import prepare as real_prepare
from kvbench_asym.analyze import group_rows


class Process:
    def __init__(self):self.pid=123456;self.returncode=None
    def poll(self):return self.returncode


class PipelineTests(unittest.TestCase):
    def run_mock(self,root,resume=False,fail_case=None):
        root=Path(root);out=root/'run';suite=root/'suite.json'
        stopped=[];requests=[];proc=Process()
        pkg=ModuleType('kvbench_longcontext');pkg.__path__=[]
        common=ModuleType('kvbench_longcontext.common');common.tokenizer_identity=lambda model:{'mock':str(model)}
        serve=ModuleType('kvbench_longcontext.serve')
        serve.launch_command=lambda args,name:['python','-m','server']
        serve.run_text=lambda *a,**k:{'returncode':0,'stdout':'','stderr':''}
        kv=ModuleType('kvbench_residual');kv.__path__=[]
        kserve=ModuleType('kvbench_residual.serve')
        kserve.stop_owned_group=lambda p:(stopped.append(p.pid),setattr(p,'returncode',0))
        trace=ModuleType('kvbench_residual.trace');trace.summarize=lambda *a,**k:{'source_verified':True,'fused_launch_observed':True}
        modules={'kvbench_longcontext':pkg,'kvbench_longcontext.common':common,'kvbench_longcontext.serve':serve,
                 'kvbench_residual':kv,'kvbench_residual.serve':kserve,'kvbench_residual.trace':trace}
        def build(path):
            path.mkdir();atomic_json(path/'runtime.json',{'mock':True})
            return {'executed_source_sha256':'mock','executed_files_sha256':{}}
        def ready(*args):
            atomic_json(out/'trace_control/hook_mock.json',{'mock':True})
            return {'mock':True}
        def request(base,name,row,s,cap,dest):
            dest.write_bytes(b'mock response')
            arm=out/'trace_control/arm.json'
            if arm.exists():
                info=read_json(arm);requests.append(row['sample_index'])
                if row['sample_index']==fail_case:raise RuntimeError('injected test-only request failure')
                td=Path(info['trace_dir']);npz=td/'owned.npz';npz.write_bytes(b'mock snapshot')
                atomic_json(td/'trace_mock.json',dict(request=info,cache_ordinal=0,decode_call_index=0,
                             arrays_file=npz.name,arrays_sha256=sha_file(npz)))
            return dict(completion_tokens=2,finish_reason='stop',protocol_valid=True)
        def analysis(paths,path,*args,**kwargs):
            path.mkdir()
            r,_,_=group_rows(group(),16,['1.1'],['0','.5','1'],['staged_mass'],[42],
                           dict(dataset='narrativeqa',trace='mock',kv_head=0))
            with gzip.open(path/'rows.csv.gz','wt',encoding='utf-8',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(r[0]));w.writeheader();w.writerows(r)
            atomic_json(path/'receipt.json',{'mock_only':True})
        def prepare(path,*args,**kwargs):
            return real_prepare(path,*args,**kwargs,tokenizer=Tokenizer())
        args=['--upstream-suite',str(suite),'--out',str(out),'--case-limit','3',
              '--layers','0','--decode-calls','0','--diagnostic-max-tokens','8','--bootstrap','10']
        if resume:args+=['--resume']
        with patch.dict('sys.modules',modules),patch('kvbench_asym.runtime.build',side_effect=build),\
             patch('kvbench_asym.run.prepare',side_effect=prepare),\
             patch('kvbench_asym.run.subprocess.run',return_value=SimpleNamespace(returncode=0)),\
             patch('kvbench_asym.run.subprocess.Popen',return_value=proc),\
             patch('kvbench_residual_diag.run.ready',side_effect=ready),\
             patch('kvbench_residual_diag.run.request_one',side_effect=request),\
             patch('kvbench_asym.analyze.analyze_paths',side_effect=analysis),\
             patch('kvbench_asym.run.socket.socket') as socket:
            socket.return_value.__enter__.return_value.connect_ex.return_value=1
            main(args)
        return requests,stopped

    def setup_data(self,root):
        root=Path(root);model=root/'model';model.mkdir()
        atomic_json(model/'config.json',dict(model_type='qwen3',num_attention_heads=32,num_key_value_heads=8,head_dim=128,num_hidden_layers=36))
        z=root/'data.zip';make_zip(z)
        atomic_json(root/'suite.json',dict(schema='kvbench.longcontext.full_suite.v1',methods=[{'state':'completed'}],
             settings=dict(model=str(model),source=str(z),data=str(root),kit=str(root),server_env='mock',max_model_len=4096,
                           batch_tokens=1024,gpu_util=.85,server_seed=42)))

    def test_complete_and_resume_no_repeat(self):
        with tempfile.TemporaryDirectory() as d:
            self.setup_data(d)
            req,stopped=self.run_mock(d)
            self.assertEqual(req,[0,1,2]);self.assertEqual(len(stopped),1)
            state=read_json(Path(d)/'run/status.json');self.assertEqual(state['status'],'completed')
            req2,stopped2=self.run_mock(d,True)
            self.assertEqual(req2,[]);self.assertEqual(stopped2,[])
            self.assertEqual(len(list((Path(d)/'run/cases').glob('*/done.json'))),3)

    def test_failed_request_resumes_only_uncommitted(self):
        with tempfile.TemporaryDirectory() as d:
            self.setup_data(d)
            with self.assertRaisesRegex(RuntimeError,'injected'):
                self.run_mock(d,fail_case=1)
            self.assertEqual(read_json(Path(d)/'run/status.json')['status'],'failed')
            req,_=self.run_mock(d,True)
            self.assertEqual(req,[1,2])
