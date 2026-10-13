"""Mock tensor/API tests for capture plumbing; these are NOT GPU executions."""
from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
from types import ModuleType,SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np
from kvbench_residual_diag.capture import Capture,ObservedLaunch
from kvbench_residual_diag.common import atomic_json,load_trace


class Tensor:
    def __init__(self,a):self.a=np.asarray(a);self.device='mock-cpu'
    @property
    def shape(self):return self.a.shape
    @property
    def ndim(self):return self.a.ndim
    @property
    def dtype(self):return self.a.dtype
    def data_ptr(self):return self.a.ctypes.data
    def __getitem__(self,i):return Tensor(self.a[i])
    def item(self):return self.a.item()
    def index_select(self,axis,indices):return Tensor(np.take(self.a,np.asarray(indices),axis=axis))
    def detach(self):return self
    def cpu(self):return self
    def numpy(self):return self.a
    def reshape(self,*shape):return Tensor(self.a.reshape(*shape))


@dataclass(frozen=True)
class Layout:
    group:int=128
    value_bits:int=4
    head_dim:int=128
    key_bits:int=4
    @property
    def offsets(self):
        return dict(K_PACKED_OFFSET=0,K_S_COL_OFFSET=8192,K_ZP_OFFSET=8448,K_S_ROW_OFFSET=8704,
                    V_PACKED_OFFSET=8960,V_S_COL_OFFSET=17152,V_S_ROW_OFFSET=17408,V_ZP_OFFSET=17664)
    @property
    def tile_bytes(self):return 17920

@dataclass
class Case:
    name:str
    seq_lens:tuple
    layout:Layout
    window:int=0
    hq:int=32
    hk:int=8


def mock_modules():
    torch=ModuleType('torch')
    torch.float16=np.dtype('float16');torch.uint8=np.dtype('uint8');torch.long=np.dtype('int64')
    torch.as_tensor=lambda x,**kw:np.asarray(x,dtype=kw.get('dtype'))
    torch.cuda=SimpleNamespace(is_current_stream_capturing=lambda:False)
    pkg=ModuleType('kvbench_residual');pkg.__path__=[]
    fixtures=ModuleType('kvbench_residual.fixtures');fixtures.Layout=Layout;fixtures.Case=Case
    return {'torch':torch,'kvbench_residual':pkg,'kvbench_residual.fixtures':fixtures}


def fixture(root):
    root=Path(root)
    source=root/'source.py';source.write_text('# mock source only\n')
    atomic_json(root/'capture_config.json',dict(layer_ordinals=[0],decode_calls=[0,15],min_context=1,max_traces=10,max_bytes=10*1024*1024))
    atomic_json(root/'arm.json',dict(epoch='test-only',dataset='test',case_id='test'))
    cache=np.arange(10*8*17920,dtype=np.uint8).reshape(10,8,17920)
    slots=np.full(10,-1,np.int32);slots[7]=1;slots[5]=2
    bt=np.full((1,256),-1,np.int32);bt[0,:3]=[7,2,5]
    args=(Tensor(np.ones((1,32,128),np.float16)),None,Tensor(bt),Tensor(np.array([300],np.int32)),Tensor(slots),Tensor(cache),
          Tensor(np.ones((3,128,8,128),np.float16)),Tensor(np.ones((3,128,8,128),np.float16)*2),
          Tensor(np.ones((32,128),np.float16)),1/np.sqrt(128))
    kw=dict(D=128,GROUP=128,K_BITS=4,V_BITS=4,SLIDING_WINDOW=0,VQ_INDIRECT=False,**Layout().offsets)
    inner=SimpleNamespace(best_config=SimpleNamespace(kwargs={'BLOCK_N':32},num_warps=4,num_stages=1,maxnreg=None))
    cap=Capture(root,source)
    return cap,args,kw,inner


class CaptureTests(unittest.TestCase):
    def test_active_pages_compacted_and_pool_preserved(self):
        with tempfile.TemporaryDirectory() as d,patch.dict('sys.modules',mock_modules()):
            cap,args,kw,inner=fixture(d)
            native_before=args[5].a.copy()
            item=cap.choose(args,kw,inner,'single');cap.save(item,args[8])
            path=next(Path(d).glob('trace_*.json'))
            data,meta=load_trace(path)
            self.assertEqual(data['cache'].shape,(3,8,17920))
            self.assertEqual(data['block_table'][0,:3].tolist(),[2,0,1])
            self.assertEqual(data['block_to_slot'].tolist(),[-1,1,0])
            self.assertEqual(data['pool_k'].shape,(2,128,8,128))
            np.testing.assert_array_equal(data['cache'][0],native_before[2])
            self.assertFalse(data['cache'][1:].any())
            np.testing.assert_array_equal(args[5].a,native_before)
            self.assertEqual(meta['active_pages'],3)
    def test_split_snapshot_waits_for_stage2(self):
        class Kernel:
            best_config=SimpleNamespace(kwargs={'BLOCK_N':32},num_warps=4,num_stages=1,maxnreg=None)
            def __getitem__(self,grid): return lambda *args,**kwargs: None
        with tempfile.TemporaryDirectory() as d,patch.dict('sys.modules',mock_modules()):
            cap,args,kw,_=fixture(d)
            mid=Tensor(np.zeros((32,4,128),np.float32))
            mid_lse=Tensor(np.zeros((32,4),np.float32))
            split_args=(*args[:8],mid,mid_lse,args[9])
            kw['NUM_KV_SPLITS']=4
            stage1=ObservedLaunch(Kernel(),'_kvarn_fused_decode_stage1',cap)
            stage2=ObservedLaunch(Kernel(),'_kvarn_fused_decode_stage2',cap)
            stage1[(1,8,4)](*split_args,**kw)
            self.assertIsNotNone(cap.pending)
            self.assertEqual(list(Path(d).glob('trace_*.json')),[])
            stage2[(32,)](mid,mid_lse,args[8])
            self.assertIsNone(cap.pending)
            path=next(Path(d).glob('trace_*.json'))
            data,meta=load_trace(path)
            self.assertEqual(meta['mode'],'split');self.assertEqual(meta['splits'],4)
            np.testing.assert_array_equal(data['reference_gpu_rot'],np.ones((1,32,128),np.float16))
    def test_real_driver_flattened_query_roundtrip(self):
        with tempfile.TemporaryDirectory() as d,patch.dict('sys.modules',mock_modules()):
            cap,args,kw,inner=fixture(d)
            q=np.arange(32*128,dtype=np.float16).reshape(32,128)/128
            args=(Tensor(q),*args[1:])
            cap.save(cap.choose(args,kw,inner,'single'),args[8])
            data,meta=load_trace(next(Path(d).glob('trace_*.json')))
            self.assertEqual(meta['original_query_shape'],[32,128])
            np.testing.assert_array_equal(data['query'],q.reshape(1,32,128))
    def test_checksum_mutation_detected(self):
        with tempfile.TemporaryDirectory() as d,patch.dict('sys.modules',mock_modules()):
            cap,args,kw,inner=fixture(d);cap.save(cap.choose(args,kw,inner,'single'),args[8])
            path=next(Path(d).glob('trace_*.json'));meta=json.loads(path.read_text())
            af=Path(d)/meta['arrays_file'];af.write_bytes(b'bad')
            with self.assertRaises(ValueError):load_trace(path)
    def test_unarmed_warmup_not_captured(self):
        with tempfile.TemporaryDirectory() as d,patch.dict('sys.modules',mock_modules()):
            cap,args,kw,inner=fixture(d);(Path(d)/'arm.json').unlink()
            self.assertIsNone(cap.choose(args,kw,inner,'single'))
    def test_indirect_verify_not_captured(self):
        with tempfile.TemporaryDirectory() as d,patch.dict('sys.modules',mock_modules()):
            cap,args,kw,inner=fixture(d);kw['VQ_INDIRECT']=True
            self.assertIsNone(cap.choose(args,kw,inner,'single'))
    def test_unknown_autotune_geometry_refused(self):
        with tempfile.TemporaryDirectory() as d,patch.dict('sys.modules',mock_modules()):
            cap,args,kw,inner=fixture(d);inner.best_config=None
            with self.assertRaises(RuntimeError):cap.choose(args,kw,inner,'single')
    def test_capture_graph_refused(self):
        modules=mock_modules();modules['torch'].cuda.is_current_stream_capturing=lambda:True
        with tempfile.TemporaryDirectory() as d,patch.dict('sys.modules',modules):
            cap,args,kw,inner=fixture(d)
            with self.assertRaises(RuntimeError):cap.choose(args,kw,inner,'single')
    def test_byte_cap_enforced_before_snapshot(self):
        with tempfile.TemporaryDirectory() as d,patch.dict('sys.modules',mock_modules()):
            cap,args,kw,inner=fixture(d);cap.config['max_bytes']=32
            item=cap.choose(args,kw,inner,'single')
            with self.assertRaises(RuntimeError):cap.save(item,args[8])
            self.assertEqual(list(Path(d).glob('trace_*.npz')),[])
    def test_epoch_resets_decode_call_index(self):
        with tempfile.TemporaryDirectory() as d,patch.dict('sys.modules',mock_modules()):
            cap,args,kw,inner=fixture(d)
            self.assertEqual(cap.choose(args,kw,inner,'single')['step'],0)
            self.assertIsNone(cap.choose(args,kw,inner,'single'))
            atomic_json(Path(d)/'arm.json',dict(epoch='second',dataset='test',case_id='test2'))
            self.assertEqual(cap.choose(args,kw,inner,'single')['step'],0)
    def test_changed_dtype_refused(self):
        with tempfile.TemporaryDirectory() as d,patch.dict('sys.modules',mock_modules()):
            cap,args,kw,inner=fixture(d)
            args=(Tensor(np.ones((1,32,128),np.float32)),*args[1:])
            with self.assertRaises(RuntimeError):cap.choose(args,kw,inner,'single')

if __name__=='__main__':unittest.main()
