"""Record completed preparation, without promoting technical checks to science."""
import hashlib
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/continuation_20261028'
OUT = RUN / 'reid_candidates_preparation'

def main():
    assert json.loads((RUN / 'queue_status.json').read_text(encoding='utf-8'))['job'] == '125_history_author_audit.json', 'Do not edit journal during catalogue checks'
    checks = {name: json.loads((OUT / name).read_text(encoding='utf-8')) for name in ['download_status.json', 'checkpoint_inspection.json', 'cpu_forward_check.json', 'real_input_check.json']}
    assert checks['real_input_check.json']['passed']
    assert not checks['real_input_check.json']['hota_measured']
    evidence = ['tracking/masa_detic_prepared.py', 'tests/test_masa_detic_prepared.py', 'scripts/check_masa_detic_real_input_20261029.py']
    result = {'prepared_at': datetime.now(timezone.utc).isoformat(), 'technical_checks': checks,
              'source_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in evidence},
              'catalogue_and_graph_tests': {'passed': 62, 'seconds': 149.84, 'log': 'checks.log'},
              'port_component_tests': {'passed': 3, 'seconds': 5.28, 'log': 'port_tests.log'},
              'official_full_pipeline_parity': False, 'hota_measured': False, 'scientific_verdict': None,
              'mgor': 'Official source snapshot prepared; trained checkpoint not found in checked sources',
              'scaletrack': 'Executable official code and weights not confirmed',
              'next': 'Validate full MASA-Detic preprocessing/feature path against reference before matched tracker experiment; do not reuse old trained heads with new embeddings'}
    target = OUT / 'preparation_result.json'
    assert not target.exists(), 'Append a new revision instead of overwriting'
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    stamp = datetime.now(timezone(timedelta(hours=3))).strftime('%d.%m.%Y, %H:%M MSK')
    text = f'''**{stamp}** — MASA-Detic: веса и отдельный путь признаков подготовлены; научная проверка ещё не выполнена

Получено. Официальный detic_masa.pth полностью сохранён и проверен по SHA256 из LFS: runs/continuation_20261028/reid_candidates_preparation/download_status.json. Отдельный tracking/masa_detic_prepared.py строго загрузил все параметры извлекателя без случайной подмены и без установки пакетов в основное окружение. Проверены компоненты переноса: port_tests.log — 3 passed за5,28с. На реальном существующем train-кадре получены7векторов размерности256 за9,109сCPU; GTнечитался, CUDAнеинициализирована, буферы неизменны: real_input_check.json. Это техническая проверка входа, не качества и не скорости GPU. Полное численное соответствие официальному пути ещё не проверено. Каталог и граф: checks.log — 62 passed за149,84с; первичный итог подготовки preparation_result.json.

MGOR: сохранён официальный снимок61файла с проверкой Git blob и SHA256, mgor/source_prepared.json; обученные веса в проверенных источниках не найдены. ScaleTrack: исполняемый официальный код и веса не подтверждены. Отсутствие в проверенных источниках не доказывает глобальное отсутствие. Не подменять эти методы случайной моделью или собственной приблизительной реализацией под прежним именем.

Решение. Т10–Т12 остаются красными, карточки planned, HOTAнеизмерена. Следующая подготовка MASA-Detic — соответствие официальной обработке/признакам, затем отдельный согласованный полный трекер с неизменными детекциями. Готовые старые головы не принимать новые признаки без проверки распределения и соответствующего обучения. Новые GPU-задания пока не опубликованы; приоритет текущей матрицы и Б2 сохранён. Полный результат всех трёх до07:40необещан.

'''
    with (OUT / 'ПОДГОТОВКА.md').open('a', encoding='utf-8') as f:
        f.write('\n\n## 6. Фактическое завершение подготовки\n\n' + text)
    with (RUN / 'HANDOFF.md').open('a', encoding='utf-8') as f:
        f.write('\n\n' + text)
    plan = ROOT / 'dissertation/ЕДИНЫЙ_ПЛАН.md'
    body = plan.read_text(encoding='utf-8')
    marker = '## Журнал изменений'
    assert body.count(marker) == 1
    plan.write_text(body.replace(marker, marker + '\n\n' + text, 1), encoding='utf-8')
    print(target)

if __name__ == '__main__':
    main()
