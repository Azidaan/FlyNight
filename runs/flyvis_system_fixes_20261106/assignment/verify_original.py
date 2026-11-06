"""Run corrected regression fixture against the immutable original module."""
from pathlib import Path
import sys,importlib.util
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
path=Path(__file__).with_name('original_flyvis_tracker.py')
spec=importlib.util.spec_from_file_location('tracking.flyvis_tracker',path)
module=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=module
spec.loader.exec_module(module)
import pytest
raise SystemExit(pytest.main(['tests/test_flyvis_assignment_contracts.py','-q','-p','no:cacheprovider']))
