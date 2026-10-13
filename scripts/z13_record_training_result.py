"""Record the completed Z13 training stage without claiming unmeasured HOTA."""
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/z13_codex_window_20261013'


def main():
    arms = {}
    for arm in ('B', 'C', 'D'):
        folder = RUN / arm
        checkpoint = torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=False)
        model = torch.load(folder / 'model.pt', map_location='cpu', weights_only=False)
        tau = json.loads((folder / 'tau.json').read_text(encoding='utf-8'))
        losses = json.loads((folder / 'loss.json').read_text(encoding='utf-8'))
        split = json.loads((folder / 'split.json').read_text(encoding='utf-8'))
        manifest = json.loads((RUN / f'{arm}_manifest.json').read_text(encoding='utf-8'))
        assert checkpoint['arm'] == model['arm'] == arm
        assert checkpoint['step'] == len(losses) == 4000
        assert [r['it'] for r in losses] == list(range(1, 4001))
        assert model['tau'] == tau == {'tau': None}
        assert len(split['fit']) == 104 and len(split['tau']) == 26
        assert checkpoint['identity'] == manifest
        assert checkpoint['optimizer']['state'] and checkpoint['rng_cuda'] is not None
        assert all(np.isfinite(r['object']) and np.isfinite(r['penalty']) for r in losses)
        arms[arm] = {
            'steps': 4000, 'checkpoint_verified': True, 'tau': None,
            'fit_videos': 104, 'threshold_videos': 26,
            'object_loss_first50': float(np.mean([r['object'] for r in losses[:50]])),
            'object_loss_last50': float(np.mean([r['object'] for r in losses[-50:]])),
            'model_sha256': hashlib.sha256((folder / 'model.pt').read_bytes()).hexdigest(),
            'source': str(folder.relative_to(ROOT)),
        }
    controller = json.loads((RUN / 'job_174009/status.json').read_text(encoding='utf-8'))
    assert controller['state'] == 'training_completed'
    report = {
        'verified_at': datetime.now().astimezone().isoformat(),
        'completed_at': controller['updated'], 'arms': arms,
        'passed_execution': True, 'working_threshold_found': False,
        'threshold_rule': 'At least 50 labelled interventions; at least 90% improve center error and remain within 0.25 object scale',
        'dev_hota_measured': False, 'heldout_val_opened': False,
        'action': 'No intervention when tau is None (tracking/z13_readout.py)',
        'decision': 'Stop this implementation after train-only threshold selection; do not run identical dev/MASA outputs',
        'limitations': 'No retained threshold predictions: cannot determine whether failure arose mainly from localization, abstention scores, minimum count, or a combination. No claim that object training or FlyVis is generally ineffective.',
    }
    result_path = RUN / 'training_result.json'
    result_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    source = str(result_path.relative_to(ROOT)).replace('\\', '/')
    path = ROOT / 'experiments/project_history.json'
    catalog = json.loads(path.read_text(encoding='utf-8'))
    card = next(e for e in catalog['experiments'] if e['id'] == 'fly_object_readout_z13_v1')
    updates = {
        'summary': 'Обучение B/C/D завершено; допустимый рабочий порог не выбран ни в одном варианте. Эта реализация останавливается без оценки dev/MASA.',
        'obtained': f'{source}, arms: B/C/D — по 4000 шагов, 104 видео обучения и 26 выбора порога. Во всех трёх model.pt и tau.json: tau=null. Завершение 13.10.2026 18:27:46 MSK (job_174009/status.json). Сохранения, оптимизатор, генераторы и конечность обучающих потерь проверены на CPU. HOTA новых вариантов не измерялась.',
        'interpretation': 'По неизменённому правилу выбора τ ни один вариант не получил допустимого порога: не найден порог с минимум 50 вмешательствами и долей успешных не меньше 90%. Это результат правила на 26 train, не оценка независимых видео. При tau=null код действия не меняет центр прогноза. Конкретный источник провала — локализация, оценка присутствия, число вмешательств или их сочетание — по сохранённым tau.json не установлен.',
        'decision': 'Остановить эту реализацию после выбора τ; не пересчитывать dev/MASA для заведомо отсутствующих вмешательств, не менять порог и голову после результата. Модели и сохранения оставить. Линия объектного обучения FlyVis не закрыта; следующая замена узла требует отдельной постановки автора.',
        'cannot_claim': 'Не измерены HOTA, D−A, D−B и D−C на dev. Нельзя заявлять пользу или вред FlyVis, бесполезность объектного обучения вообще либо установленную причину провала. Уменьшение обучающей потери — не доказательство переноса и пользы стенду.',
    }
    for field, value in updates.items():
        old = card.get(field)
        if old and old != value:
            card.setdefault('revisions', []).append({'date': '2026-10-13', 'commit': '7252f0d', 'field': field, 'old': old, 'reason': 'Запись проверенного итога обучения и выбора τ; HOTA не измерялась'})
        card[field] = value
    card['state'] = 'completed_negative'
    card['done'] = '2026-10-13'
    if not any(e.get('ref') == source for e in card['evidence']):
        card['evidence'].append({'kind': 'json', 'ref': source, 'locator': 'arms, working_threshold_found', 'claim': 'Все три варианта завершили 4000 шагов; рабочий τ отсутствует'})
    path.write_text(json.dumps(catalog, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    plan = ROOT / 'dissertation/ЕДИНЫЙ_ПЛАН.md'
    text = plan.read_text(encoding='utf-8')
    entry = '''**13.10.2026, 19:33 MSK** — З13: обучение B/C/D закончено, рабочего порога нет; эта реализация остановлена

*Получено (`fly_object_readout_z13_v1`).* `runs/z13_codex_window_20261013/training_result.json`, `B/C/D/tau.json`, `B/C/D/model.pt`: все три варианта завершили по 4000 шагов; 104 видео использованы для обучения, 26 для выбора τ. У B, C и D τ=null. Контроллер закончил в 18:27:46 (`job_174009/status.json`), до временного предела и без незавершённых заданий. Проверены сохранения шага 4000, оптимизатор, генераторы, последовательность шагов, конечность потерь и согласование τ с экспортированными моделями.

*Вывод.* По заранее заданному правилу на train нет рабочего τ с минимум 50 вмешательствами и не меньше 90% успешных поправок. При τ=null `tracking/z13_readout.py` не меняет центр прогноза. Это отрицательный итог этой реализации, не доказательство бесполезности объектного обучения FlyVis. Предсказания выбора τ не сохранены, поэтому конкретная причина — локализация, вероятность присутствия, число вмешательств или сочетание — не установлена. Основные критерии HOTA остаются неизмеренными.

*Решение.* Не запускать дорогую сборку dev/MASA для отсутствующего действия; настройки после результата не менять. Модели и состояния сохранены, отложенные 75 val не использованы. Сводка к стенду: новых измерений HOTA нет, действия модели отключены по протоколу; процент измеренного улучшения и доля оракула не оценены.

'''
    marker = '## Журнал изменений\n\n'
    assert marker in text
    if entry.splitlines()[0] not in text:
        plan.write_text(text.replace(marker, marker + entry, 1), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
