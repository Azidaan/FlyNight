"""Exercise missing prerequisites without loading FlyVis or changing user data."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import pytest
import torch
import tracking.flyvis_adapter as adapter

source = ROOT/'tests/test_flyvis_activity_readout.py'
before = OUT/'original_test_flyvis_activity_readout.py'
name = 'test_estimator_exposes_internal_activity_of_current_frame_only'


def assertions(path):
    function = next(node for node in ast.parse(path.read_text(encoding='utf-8')).body
                    if isinstance(node, ast.FunctionDef) and node.name == name)
    return [ast.dump(node, include_attributes=False) for node in ast.walk(function)
            if isinstance(node, ast.Assert)]


assert assertions(source) == assertions(before)
spec = importlib.util.spec_from_file_location('activity_guard_tests', source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
test = getattr(module, name)
runtime_before, data_before = adapter.DEFAULT_RUNTIME_DIR, adapter.DEFAULT_DATA_DIR
rows = []
with tempfile.TemporaryDirectory(prefix='guard_', dir=OUT) as scratch:
    scratch = Path(scratch)
    try:
        adapter.DEFAULT_RUNTIME_DIR = scratch/'runtime'
        adapter.DEFAULT_DATA_DIR = scratch/'data'
        for condition in ('missing_runtime', 'missing_checkpoint'):
            if condition == 'missing_checkpoint':
                (adapter.DEFAULT_RUNTIME_DIR/'flyvis').mkdir(parents=True)
            try:
                test()
            except pytest.skip.Exception as error:
                rows.append(dict(condition=condition, skipped=True, reason=str(error)))
            else:
                raise AssertionError('Missing prerequisite was not skipped: '+condition)
    finally:
        adapter.DEFAULT_RUNTIME_DIR, adapter.DEFAULT_DATA_DIR = runtime_before, data_before
assert 'flyvis' not in sys.modules
assert not torch.cuda.is_initialized()
result = dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
              original_sha256=hashlib.sha256(before.read_bytes()).hexdigest(),
              original_assertions_preserved=True, assertions=len(assertions(source)),
              cuda_initialized=False, vendor_imported=False, conditions=rows)
(OUT/'guard_report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print(json.dumps(result, ensure_ascii=False, indent=2))
