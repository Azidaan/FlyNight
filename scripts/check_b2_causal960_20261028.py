"""Проверки зарегистрированной новой реализации, затем один GPU-профиль."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
import sys, subprocess
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.continuation_service_20261028 import RUN, service


def main():
    folder = RUN / 'b2_causal960_checks'
    for i, args in enumerate([['scripts/validate_project_history.py'],
                             ['-m', 'pytest', 'tests/test_project_history.py', 'tests/test_b2_causal_detector.py', '-q']]):
        with (folder / f'check_{i}.log').open('w', encoding='utf-8') as out:
            code = subprocess.run([sys.executable, *args], cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, timeout=420).returncode
        assert code == 0
    service.write(folder / 'report.json', dict(passed=True, updated=service.now()))
    script = 'scripts/profile_b2_causal960_20261028.py'
    target = RUN / 'b2_causal960_profile'
    service.write(RUN / 'ready_jobs/113_b2_causal960_profile.json', dict(name='b2_causal960_profile',
                  folder=target.relative_to(ROOT).as_posix(), script=script,
                  script_sha256=service.sha(ROOT / script), protocol_sha256=service.sha(target / 'protocol.json')))


if __name__ == '__main__': main()
