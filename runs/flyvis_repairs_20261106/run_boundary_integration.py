"""Run the existing bounded audit suite plus the video-identity regressions."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts import run_system_fix_checks_20261106 as suite

if __name__ == '__main__':
    suite.TESTS = [*suite.TESTS, 'test_motion_cache_video_identity.py',
                   'test_b1_timestamped_optimizer_numpy.py']
    assert len(suite.TESTS) == len(set(suite.TESTS))
    suite.main()
