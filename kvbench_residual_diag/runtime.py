"""Append a capture observer only to a newly created, pinned Parent runtime."""
from __future__ import annotations
from pathlib import Path
import shutil
from .common import atomic_json,sha_file


def build(out):
    from kvbench_residual.runtime import build_runtime,verify_runtime,TARGET
    out=Path(out).resolve()
    manifest=build_runtime(out,"parent",epsilon=.02)
    source=Path(__file__).resolve().parent
    shutil.copytree(source,out/"kvbench_residual_diag",ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    kernel=out/"vllm"/TARGET
    text=kernel.read_text(encoding="utf-8")
    footer=("\n# EAGER-ONLY real activation capture, not performance measurement.\n"
            "from kvbench_residual_diag.capture import install as _ar_diag_install\n"
            "_ar_diag_install(globals(), __file__)\n")
    compile(text+footer,str(kernel),"exec")
    kernel.write_text(text+footer,encoding="utf-8")
    executed=sha_file(kernel)
    manifest["executed_files_sha256"][str(TARGET)]=executed
    manifest["executed_source_sha256"]=executed
    manifest["files_sha256"]={str(p.relative_to(out)):sha_file(p) for p in sorted(out.rglob("*"))
                              if p.is_file() and not p.is_symlink() and p != out/"runtime.json"}
    manifest["diagnostic_capture"]={"schema":"kvbench.residual.capture_runtime.v1", "eager_required":True,
                                    "parent_arithmetic_unchanged":True,"production_timing_valid":False,
                                    "diagnostic_package_hashes":{str(p.relative_to(out)):sha_file(p)
                                                               for p in sorted((out/"kvbench_residual_diag").rglob("*.py"))}}
    manifest["traffic_note"]="Real input snapshots only. D2H synchronized. No production timing or DRAM claim."
    atomic_json(out/"runtime.json",manifest)
    verify_runtime(out,"parent",full=True,epsilon=.02)
    return manifest
