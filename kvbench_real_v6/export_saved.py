#!/usr/bin/env python3
"""Re-export an interrupted/measured point without model inference."""
import argparse,json
from pathlib import Path
from outputs import export_run
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('attempt',type=Path)
a=p.parse_args()
state=json.loads((a.attempt/'state.json').read_text())
expected=json.loads((a.attempt/'measure.manifest.json').read_text())
print(json.dumps(export_run(a.attempt/'measured',a.attempt,expected,state['point']['thinking']),ensure_ascii=False,indent=2))
