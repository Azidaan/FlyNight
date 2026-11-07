"""Обязательные проверки каталога перед публикацией единственного GPU-этапа."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '2'
from pathlib import Path
import subprocess, sys, time
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.recovery_research_service_20261027 import RUN, read, write, sha, now


def main():
    folder = RUN / 'b1_tracker_checks'
    p = read(folder / 'protocol.json')
    for path, h in p['sha256'].items():
        assert sha(ROOT / path) == h
    began = time.monotonic()
    outcomes = []
    commands = [['scripts/validate_project_history.py'],
                ['-m', 'pytest', 'tests/test_project_history.py', '-q'],
                ['-m', 'pytest', 'tests/test_b1_tracker_runtime.py', '-q']]
    for i, command in enumerate(commands):
        write(folder / 'status.json', dict(state='running', stage=command, updated=now()))
        log = folder / ('check' + str(i) + '.log')
        with log.open('w', encoding='utf-8') as stream:
            result = subprocess.run([sys.executable, *command], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW, timeout=180)
        outcomes.append(dict(command=command, exit_code=result.returncode, log=log.relative_to(ROOT).as_posix(), sha256=sha(log)))
        assert result.returncode == 0, outcomes[-1]
    write(RUN / 'b1_tracker_catalog_checks.json', dict(passed=True, outcomes=outcomes, seconds=time.monotonic() - began, updated=now()))
    from scripts.prepare_b1_tracker_20261028 import publish
    publish()
    write(folder / 'report.json', dict(passed=True, published=['270_b1_full_tracker', '275_b1_full_tracker_audit'], updated=now()))


if __name__ == '__main__':
    main()
