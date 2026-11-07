"""Принять карточку ценового стопа и сохраняющий фундаментальный обзор."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '2'
from pathlib import Path
import sys, subprocess, time
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.recovery_research_service_20261027 import RUN, read, write, now, sha


def main():
    folder = RUN / 'b1_result_checks'; start = time.monotonic()
    p = read(folder / 'protocol.json')
    for path, h in p['sha256'].items(): assert sha(ROOT / path) == h, path
    results = []
    commands = [['scripts/validate_project_history.py'], ['-m', 'pytest', 'tests/test_project_history.py', '-q'],
                ['scripts/restore_obsidian_sections_20261027.py'], ['-m', 'pytest', 'tests/test_obsidian_fundamental_overview.py', '-q']]
    for i, command in enumerate(commands):
        write(folder / 'status.json', dict(state='running', stage=command, updated=now()))
        log = folder / ('check' + str(i) + '.log')
        with log.open('w', encoding='utf-8') as stream:
            c = subprocess.run([sys.executable, *command], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW, timeout=210)
        results.append(dict(command=command, exit_code=c.returncode, log=log.relative_to(ROOT).as_posix(), sha256=sha(log)))
        assert c.returncode == 0, results[-1]
    write(folder / 'report.json', dict(passed=True, results=results, seconds=time.monotonic() - start, updated=now()))
    path = RUN / 'HANDOFF.md'
    path.write_text(path.read_text(encoding='utf-8')+'\n\n310b1_result_checks завершила каталог/pytest/сохраняющий граф/3теста графа, всеcode0. Первичный b1_result_checks/report.json. Б1первыйполныйпроходаудирован, полнаясериязакрытапостоимости, не научныйитогБ1. Следующийэтап требуетагента: Б2новаянаучнаяподготовка, не старый3-stepповтор.\n', encoding='utf-8')


if __name__ == '__main__': main()
