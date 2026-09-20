"""Integration of measurement/export orchestration with mocked GPU/HTTP/EvalScope boundaries."""
import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from test_protocol import config,data,make_db
from run_real import NaturalRunner,plan_points,recommend
from engine import ServerManager

class NullTelemetry:
    def __init__(self,*a):pass
    def start(self):pass
    def stop(self):pass

class Tokenizer:
    def apply_chat_template(self,*a,**kw):return [1]*20

class TestWorkflow(unittest.TestCase):
    def test_one_natural_point_exports_complete_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);cfg=config();cfg.update(run_dir=str(root),root=str(root),model='mock-model',
                port=8000,gpu='0',sample_interval=2,settle_seconds=0,request_timeout=60,data_seed=42,
                stage='pilot',preset='real',pilot_size=8,concurrency=[2])
            rows={'math_500':data(n=500),'humaneval':data('humaneval',164)}
            p=plan_points(cfg,rows)[0]
            runner=NaturalRunner.__new__(NaturalRunner);ServerManager.__init__(runner,cfg)
            runner.rows=rows;runner.tokenizer=Tokenizer()
            def start(point):
                runner.server_folder=root/'servers'/'mock';runner.server_folder.mkdir(parents=True,exist_ok=True)
                (runner.server_folder/'launcher.log').write_text('GPU KV cache size: 49,152 tokens\n')
            def invoke(cmd,logfile):
                self.assertIn('--stream',cmd);self.assertNotIn('--min-tokens',cmd)
                path=Path(cmd[cmd.index('--dataset-path')+1]);out=Path(cmd[cmd.index('--outputs-dir')+1])
                req=[{'request':json.loads(x)} for x in path.read_text().splitlines()]
                self.assertTrue(all(r['request']['ignore_eos'] is False for r in req))
                make_db(out,req)
                with (runner.server_folder/'launcher.log').open('a') as f:f.write('Prefix cache hit rate: 1.2%\n')
            runner.start=start;runner.invoke=invoke
            d=root/'points'/p['id']/'attempt001';d.mkdir(parents=True)
            with patch('run_real.Telemetry',NullTelemetry),patch('run_real.http_text',return_value='vllm:num_preemptions_total 0\n'):
                runner.one(p,d)
            state=json.loads((d/'state.json').read_text());s=json.loads((d/'stats.json').read_text())
            self.assertEqual(state['status'],'ok');self.assertTrue(s['measurement_complete'])
            self.assertEqual(s['kv_capacity_tokens_single_gpu_log'],49152)
            self.assertEqual(s['prefix_native_last_percent'],1.2)
            self.assertTrue((d/'requests.jsonl').exists());self.assertTrue((d/'native_prefix_stats.json').exists())
            self.assertTrue((root/'summary.csv').exists())

if __name__=='__main__':unittest.main()
