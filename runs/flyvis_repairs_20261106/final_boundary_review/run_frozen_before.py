"""Run final regressions against the frozen original module without editing it."""
from pathlib import Path
import importlib.util
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import scripts
import pytest

path = Path(__file__).resolve().parent / "baseline/run_bdd_video_rate_experiment.py"
name = "scripts.run_bdd_video_rate_experiment"
spec = importlib.util.spec_from_file_location(name, path)
module = importlib.util.module_from_spec(spec)
sys.modules[name] = module
setattr(scripts, "run_bdd_video_rate_experiment", module)
spec.loader.exec_module(module)
status = pytest.main([str(ROOT / "tests/test_motion_cache_video_identity.py"), "-q"])
import torch
assert not torch.cuda.is_initialized()
raise SystemExit(status)
