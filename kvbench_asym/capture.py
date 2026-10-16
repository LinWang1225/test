"""Bounded per-request capture using the already reviewed native trace hook."""
from __future__ import annotations
import os
from pathlib import Path
from kvbench_residual_diag.capture import Capture, ObservedLaunch
from .common import read_json


class RollingCapture(Capture):
    def __init__(self, root, source):
        super().__init__(root, source)
        self.control_root = self.root
        self.run_root = self.control_root.parent

    def arm(self):
        path = self.control_root/'arm.json'
        if not path.exists():
            return None
        arm = read_json(path)
        if arm['epoch'] != self.epoch:
            target = Path(arm['trace_dir']).resolve(strict=True)
            if not target.is_relative_to(self.run_root/'cases') or target.name != 'traces':
                raise RuntimeError('Trace target must be a newly prepared case trace directory')
            self.epoch = arm['epoch']
            self.layers, self.calls = {}, {}
            self.pending = None
            self.saved = self.bytes = 0  # caps are per request, not lifetime totals
            self.root = target
        return arm


def install(namespace, source):
    root = os.environ.get('AR_DIAG_TRACE_ROOT')
    if not root or namespace.get('AR_POLICY_NAME') != 'parent':
        raise RuntimeError('Equal-payload capture requires the isolated native Parent runtime')
    capture = RollingCapture(root, source)
    for name in ('_kvarn_fused_decode_kernel','_kvarn_fused_decode_stage1','_kvarn_fused_decode_stage2'):
        if name not in namespace:
            raise RuntimeError('Missing reviewed native kernel: '+name)
        namespace[name] = ObservedLaunch(namespace[name], name, capture)
