"""Private native-Parent overlay. No write to installed vLLM or old AR runtime."""
from __future__ import annotations
import importlib
from pathlib import Path
import shutil
from .common import atomic_json, sha_file


def check_dependencies():
    pins = __import__('kvbench_asym.dependency_pins', fromlist=['PINS']).PINS
    actual = {}
    for module, expected in pins.items():
        spec = __import__('importlib.util', fromlist=['find_spec']).find_spec(module)
        if spec is None or not spec.origin:
            raise RuntimeError('Apply the existing residual and two-experiment packages first: '+module)
        actual[module] = sha_file(spec.origin)
        if actual[module] != expected:
            raise RuntimeError('Reviewed diagnostic dependency changed: '+module+'; do not bypass the hash check. Port and revalidate explicitly.')
    # v1 handles its own six writer/reader source pins during build_runtime.
    return actual


def build(out):
    from kvbench_residual.runtime import build_runtime, verify_runtime, TARGET
    from kvbench_residual_diag import __file__ as diag_file
    out = Path(out).resolve()
    manifest = build_runtime(out, 'parent', epsilon=.02)
    for source in (Path(diag_file).parent, Path(__file__).resolve().parent):
        shutil.copytree(source, out/source.name, ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    kernel = out/'vllm'/TARGET
    footer = '\n# Isolated equal-payload diagnostic capture; NOT performance instrumentation.\nfrom kvbench_asym.capture import install as _asym_install\n_asym_install(globals(), __file__)\n'
    text = kernel.read_text(encoding='utf-8') + footer
    compile(text, str(kernel), 'exec'); kernel.write_text(text, encoding='utf-8')
    actual = sha_file(kernel)
    manifest['executed_files_sha256'][str(TARGET)] = actual
    manifest['executed_source_sha256'] = actual
    manifest['files_sha256'] = {str(p.relative_to(out)):sha_file(p) for p in sorted(out.rglob('*'))
                               if p.is_file() and not p.is_symlink() and p != out/'runtime.json'}
    manifest['equal_payload_capture'] = dict(eager_required=True, parent_representation='native_int4',
               parent_arithmetic_unchanged=True, selectors_injected_into_generation=False,
               scope='New observer only; offline comparison does not change generated trajectories.')
    atomic_json(out/'runtime.json', manifest)
    verify_runtime(out, 'parent', full=True, epsilon=.02)
    return manifest
