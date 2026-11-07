"""Зафиксировать только реально завершённую проверку исполнения Б2."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
import sys, subprocess
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.continuation_service_20261028 import RUN, service


def main():
    source = RUN / 'b2_causal960_profile'; folder = RUN / 'b2_causal960_record'
    controller = service.read(source / 'controller_result.json')
    assert controller['exit_code'] == 0 and controller['all_owned_ended']
    assert not any(service.matching(k, v) for k, v in controller['owned'].items())
    report = service.read(source / 'report.json'); assert report['passed'] and report['complete']
    history = service.read(ROOT / 'experiments/project_history.json')
    card = next(c for c in history['experiments'] if c['id'] == 'b2_causal960_path_v1')
    card.update(state='infrastructure', done='2026-10-29',
        obtained='runs/continuation_20261028/b2_causal960_profile/report.json: полный проход одного исторического train-видео, '
                 'штатные обнаружения 960 и нулевая поправка совпали с сохранённым кэшем. '
                 'Числа query/клипов/памяти/времени и производных отдельно записаны в full_video/controls.',
        interpretation='Проверен новый путь исполнения с реальными временами, кольцом признаков и отдельно обучаемыми контролями последнего/среднего. '
                       'Проверка производных не является обучением или оценкой качества. Старые три scratch-шага не повторены.',
        decision='Зарегистрировать сопоставимое обучение C/D/R1/B/last/mean, точные начальные веса, разбиения, эпохи и фактическую цену '
                 'полной train+dev эпохи до допуска серии. Если автор разрешит матрицу истории, приоритет GPU пересмотреть на сохранённой границе.',
        cannot_claim='Нет обученных весов Б2, точности детектора на val или HOTA. Проверка одного train-видео не доказывает '
                     'генерализацию/полезность ядра. Совмещение объектов в P3 не реализовано; last/mean не чистый контроль порядка.')
    for item in card['acceptance']: item.update(met=True, evidence='runs/continuation_20261028/b2_causal960_profile/report.json')
    service.write(ROOT / 'experiments/project_history.json', history)
    from scripts.b123_safe_journal_20261027 import journal
    journal(card['id'], 'Полный причинный проход Б2 на 960 с нулевым контролем',
            card['obtained'] + ' ' + card['interpretation'] + ' ' + card['cannot_claim'])
    for i, args in enumerate([['scripts/validate_project_history.py'],
                             ['-m', 'pytest', 'tests/test_project_history.py', 'tests/test_b2_causal_detector.py', '-q']]):
        with (folder / f'check_{i}.log').open('w', encoding='utf-8') as out:
            code = subprocess.run([sys.executable, *args], cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, timeout=420).returncode
        assert code == 0
    service.write(folder / 'report.json', dict(passed=True, source_sha256=service.sha(source / 'report.json'),
                  actual_tree_ended=True, catalogue_checked=True, scientific_verdict=None, updated=service.now()))


if __name__ == '__main__': main()
