"""Run the final regression cases against the immutable pre-fix estimator."""
from pathlib import Path
import hashlib
import importlib.util
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
SOURCE=Path(__file__).with_name('flyvis_motion.py')
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()=='02f025826ca3b298acfb71e7bfce2725caa9d0d862f4e696c50e537622023c38'
spec=importlib.util.spec_from_file_location('tracking.flyvis_motion',SOURCE)
module=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=module
spec.loader.exec_module(module)
import pytest
raise SystemExit(pytest.main([str(ROOT/'tests/test_flyvis_observation_transaction.py'),'-q','-p','no:cacheprovider',
    '--basetemp='+str(SOURCE.parent/'final_cases_before_temp')]))
