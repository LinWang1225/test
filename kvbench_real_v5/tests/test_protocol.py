import base64,gzip,json,os,pickle,sqlite3,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from outputs import decode_saved,export_run,join_response,key
from metrics import native_metrics,extract_server_log
from run_real import build_requests,eligible,plan_points,recommend


def config():
    return dict(methods=['fp16','tq4','kvarn'],datasets=['math_500','humaneval'],stage='pilot',pilot_size=32,
        repeats=1,apc=[0],concurrency=[2,4,8],max_seqs=16,max_len=32768,batch_tokens=1024,
        math_max_tokens=16384,human_max_tokens=8192,kv_bytes=None,gpu_util=.85,thinking=True,
        temperature=.6,top_p=.95,top_k=20,seed=42,data_seed=42,eager=False,prefix_sha256=None,
        prefix_text=None,shared_prefix_file=None,recommendations=None,warmup_requests=2,
        max_pilot_cap_fraction=.05,max_preemptions=-1,max_p95_ttft_ms=None,max_p95_tpot_ms=None)


def data(name='math_500',n=3):
    return [{'dataset':name,'case_id':f'{name}/{i}','messages':[{'role':'user','content':f'problem {i}'}],
             'reference':'SECRET_ANSWER','original_record':{'problem':f'problem {i}','answer':'SECRET_ANSWER'}} for i in range(n)]


def make_db(root,records,finish='stop',omit_last=False):
    root.mkdir(parents=True,exist_ok=True)
    con=sqlite3.connect(root/'benchmark_data.db')
    con.execute('''CREATE TABLE result(request TEXT,response_messages TEXT,success INTEGER,start_time REAL,
    completed_time REAL,latency REAL,first_chunk_latency REAL,time_per_output_token REAL,
    completion_tokens INTEGER,prompt_tokens INTEGER,inter_token_latencies TEXT,request_id TEXT)''')
    for i,r in enumerate(records[:-1] if omit_last else records):
        n=7+i
        chunks=[{'id':f'id{i}','choices':[{'index':0,'delta':{'content':'thinking中文</think>\\boxed{42}'},'finish_reason':finish}]},
                {'choices':[],'usage':{'prompt_tokens':25,'completion_tokens':n,'prompt_tokens_details':{'cached_tokens':12}}}]
        saved=base64.b64encode(pickle.dumps(chunks)).decode()
        con.execute('INSERT INTO result VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',(json.dumps(r['request']),saved,1,
            100+i,102+i,2,.2,1.8/(n-1),n,25,json.dumps([.4,.6]),f'id{i}'))
    con.commit();con.close()


class Tests(unittest.TestCase):
    def test_natural_request_no_answer_leak(self):
        cfg=config();p=plan_points(cfg,{'math_500':data(n=500),'humaneval':data('humaneval',164)})[0]
        r=build_requests(data(n=32),p,cfg)
        self.assertFalse(r[0]['request']['ignore_eos']);self.assertEqual(r[0]['request']['min_tokens'],0)
        self.assertEqual(r[0]['request']['stop'],[]);self.assertEqual(len(r),32)
        self.assertNotIn('SECRET_ANSWER',json.dumps(r[0]['request']))
        q=dict(p,method='kvarn');self.assertEqual(r,build_requests(data(n=32),q,cfg))

    def test_full_is_full_and_metadata(self):
        cfg=config();cfg.update(stage='full',apc=[0,1],concurrency=[4],kv_bytes=6*1024**3)
        points=plan_points(cfg,{'math_500':data(n=500),'humaneval':data('humaneval',164)})
        self.assertEqual(len(points),12)
        self.assertEqual(points[0]['number'],500);self.assertEqual(points[2]['number'],164)
        self.assertEqual(points[0]['kv_cache_memory_bytes_requested'],6*1024**3)
        self.assertIsNone(points[0]['evalscope_random_prefix_length'])
        self.assertIsNone(points[0]['independent_prefix_cache_bytes'])

    def test_warmup_no_matching_prefix(self):
        c=config();p=plan_points(c,{'math_500':data(n=500),'humaneval':data('humaneval',164)})[0]
        normal=build_requests(data(n=32),p,c)
        warm=build_requests(data(n=32),p,c,warm=True)
        self.assertEqual(len(warm),2)
        self.assertEqual(warm[0]['request']['messages'][0]['role'],'system')
        self.assertNotEqual(normal[0]['request']['messages'][0],warm[0]['request']['messages'][0])
        self.assertFalse(warm[0]['request']['ignore_eos'])

    def test_context_not_silently_truncated(self):
        c=config();c['max_len']=16384
        p={'number':2,'id':'x','repeat':1,'max_tokens':16384}
        class Tok:
            def apply_chat_template(self,*a,**kw):return [1]*30
        with self.assertRaises(ValueError):build_requests(data( n=2),p,c,Tok())

    def test_response_channels(self):
        chunks=[{'choices':[{'delta':{'reasoning_content':'推理'},'index':0}]},
                {'choices':[{'delta':{'content':'答案'},'index':0,'finish_reason':'stop'}]},
                {'usage':{'prompt_tokens':20,'completion_tokens':7}}]
        r=join_response(chunks,True)
        self.assertEqual(r['reasoning_content'],'推理');self.assertEqual(r['final_answer_for_scoring'],'答案')
        self.assertEqual(r['finish_reason'],'stop')

    def test_truncated_reasoning_not_final(self):
        r=join_response([{'choices':[{'delta':{'content':'reasoning with \\boxed{0}'},'finish_reason':'length'}]}],True)
        self.assertEqual(r['final_answer_for_scoring'],'');self.assertEqual(r['finish_reason'],'length')

    def test_non_thinking_code_untouched(self):
        r=join_response([{'choices':[{'delta':{'content':'def f():\n    return 1'},'finish_reason':'stop'}]}],False)
        self.assertEqual(r['final_answer_for_scoring'],'def f():\n    return 1')

    def test_safe_pickle(self):
        class Attack:
            def __reduce__(self):return (os.system,('echo MUST_NOT_RUN',))
        with self.assertRaises(ValueError):decode_saved(base64.b64encode(pickle.dumps([Attack()])).decode())
        self.assertEqual(decode_saved(base64.b64encode(pickle.dumps([{'x':'中'}])).decode()),[{'x':'中'}])

    def test_export_variable_length_and_raw(self):
        c=config();p={'number':3,'id':'test','repeat':1,'max_tokens':100}
        req=build_requests(data(),p,c)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);make_db(root/'measured',req)
            s=export_run(root/'measured',root,req,True)
            self.assertTrue(s['measurement_complete']);self.assertEqual(s['output_tokens_avg'],8)
            recs=[json.loads(x) for x in (root/'requests.jsonl').read_text().splitlines()]
            self.assertEqual(len(recs),3);self.assertEqual(recs[0]['case_id'],'math_500/0')
            self.assertIn('中文',recs[0]['raw_content']);self.assertEqual(recs[0]['final_answer_for_scoring'],'\\boxed{42}')
            with gzip.open(root/recs[0]['raw_response_file'],'rt') as f:raw=json.load(f)
            self.assertEqual(raw[-1]['usage']['completion_tokens'],7)
            self.assertEqual(recs[0]['usage']['prompt_tokens_details']['cached_tokens'],12)

    def test_truncation_retained(self):
        req=build_requests(data(),{'number':3,'id':'x','repeat':1,'max_tokens':8},config())
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);make_db(root/'measured',req,finish='length')
            s=export_run(root/'measured',root,req,True)
            self.assertTrue(s['measurement_complete']);self.assertEqual(s['length_capped'],3)
            self.assertFalse(eligible(s,config())[0])

    def test_missing_rows_not_success(self):
        req=build_requests(data(),{'number':3,'id':'x','repeat':1,'max_tokens':8},config())
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);make_db(root/'measured',req,omit_last=True)
            s=export_run(root/'measured',root,req,True)
            self.assertFalse(s['measurement_complete']);self.assertEqual(s['missing_case_ids'],['math_500/2'])

    def test_native_cache_not_estimated(self):
        a='vllm:prefix_cache_queries_total{model="x"} 10\nvllm:prefix_cache_hits_total{model="x"} 5\nvllm:num_preemptions_total 0\n'
        b=a.replace(' 10',' 20').replace(' 5',' 7')
        r=native_metrics(a,b)
        self.assertNotIn('prefix_hit_ratio',r);self.assertEqual(r['preemptions_delta'],0)
        self.assertEqual(len(r['native_cache_before']),2)

    def test_capacity_and_native_log(self):
        r=extract_server_log('GPU KV cache size: 30,720 tokens\nAvailable KV cache memory: 6.00 GiB\nPrefix cache hit rate: 12.3%\n')
        self.assertEqual(r['kv_capacity_tokens_single_gpu_log'],30720)
        self.assertEqual(r['prefix_native_last_percent'],12.3)
        self.assertFalse(r['prefix_native_last_is_run_aggregate'])

    def test_concurrency_slo_not_oom_only(self):
        c=config();c['max_p95_tpot_ms']=100
        s={'measurement_complete':True,'length_capped_fraction':0,'tpot_ms_p95':101}
        self.assertFalse(eligible(s,c)[0])
        c['max_p95_tpot_ms']=None;self.assertTrue(eligible(s,c)[0])

    def test_method_server_group_separates_apc(self):
        import inspect
        from engine import ServerManager
        self.assertIn("p['apc']",inspect.getsource(ServerManager.start).split('if self.server is not None')[0])

if __name__=='__main__':unittest.main()
