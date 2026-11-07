"""Каталог перед разрешённым продолжением; публикация ровно одной пары заданий."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '2'
from pathlib import Path
import subprocess, sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.recovery_research_service_20261027 import RUN, read, write, sha, now


def main():
    folder = RUN / 'b1_continuation_checks'; p = read(folder / 'protocol.json')
    for path, digest in p['sha256'].items(): assert sha(ROOT / path) == digest, path
    results = []
    for i, command in enumerate([['scripts/validate_project_history.py'], ['-m', 'pytest', 'tests/test_project_history.py', '-q'],
                                 ['-m', 'pytest', 'tests/test_b1_tracker_continuation.py', '-q']]):
        write(folder / 'status.json', dict(state='running', stage=command, updated=now()))
        log = folder / ('check' + str(i) + '.log')
        with log.open('w', encoding='utf-8') as stream:
            result = subprocess.run([sys.executable, *command], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW, timeout=180)
        results.append(dict(command=command, exit_code=result.returncode, log=log.relative_to(ROOT).as_posix()))
        assert result.returncode == 0, results[-1]
    write(folder / 'report.json', dict(passed=True, results=results, updated=now()))
    from scripts.prepare_b1_tracker_continuation1_20261028 import publish
    publish()


if __name__ == '__main__': main()
