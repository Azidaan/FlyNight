"""Preregister the author-requested compiler change before any measurements."""
from datetime import datetime
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tracking.z13_checkpoint import atomic_json
RUN = ROOT / 'runs/z13_triton_hota_v1_20261014'
OLD = ROOT / 'runs/z13_spatial_hota_v1_20261014'


def main():
    RUN.mkdir(exist_ok=True)
    now = datetime.now().astimezone()
    assert json.loads((OLD / 'status.json').read_text(encoding='utf-8'))['state'] == 'paused'
    auth = json.loads((OLD / 'full_authorization.json').read_text(encoding='utf-8'))
    spec = {'id': 'z13_triton_execution_v1', 'date': now.isoformat(),
            'request': 'А можешь сейчас написать по быстрому, стопнем нынешний прогон, и запустим на Тритоне?',
            'deadline': auth['deadline'], 'baseline_root': str(OLD),
            'completed_original_signature_videos': 6,
            'changes': 'Separate inference adapter: compiled HexGRU step B and FlyVis tensor Euler step C/D. Tracker/crops/readout/MASA/weights/thresholds fixed.',
            'gates': {'activation_atol': 1e-4, 'activation_rtol': 1e-4,
                      'probability_atol': 1e-5, 'probability_rtol': 1e-5,
                      'same_argmax_and_threshold_decisions': True,
                      'full_video_track_ids_equal': True, 'full_video_output_atol_px': 1e-3,
                      'full_video_wall_speedup_required': 1.0},
            'probe_gpu_limit_seconds': 1200,
            'scope': 'One compiler implementation, no training/data download/heldout data. Full run only after technical parity and speed gates; deadline unchanged. Preserve old results separately.'}
    atomic_json(RUN / 'protocol.json', spec)
    atomic_json(RUN / 'status.json', {'state': 'preparing', 'updated': now.isoformat()})
    catpath = ROOT / 'experiments/project_history.json'
    cat = json.loads(catpath.read_text(encoding='utf-8'))
    original = next(c for c in cat['experiments'] if c['id'] == 'fly_spatial_hota_policy_z13_v1')
    assert not any(c['id'] == spec['id'] for c in cat['experiments'])
    cat['experiments'].append({'id': spec['id'], 'title': 'З13 — отдельное выполнение через Triton',
         'area': original['area'], 'state': 'planned', 'started': now.strftime('%Y-%m-%d'),
         'question': 'Ускоряет ли отдельная компиляция исполнение З13 при сохранении его результатов?',
         'hypothesis': 'Объединение тензорных операций может сократить время; основной источник времени ещё не измерен.',
         'expected': 'Проверка совпадения последовательных состояний, вероятностей и решений, затем времени одного одинакового видео.',
         'acceptance': [{'criterion': 'Последовательное считывание B/C/D сохраняет состояние и вероятности в допуске, максимумы и решения порога совпадают.', 'met': None},
                        {'criterion': 'Парное видео сохраняет номера треков и рамки в допуске 1e-3 px и исполняется быстрее после компиляции.', 'met': None}],
         'evidence': [{'kind': 'json', 'ref': 'runs/z13_triton_hota_v1_20261014/protocol.json',
                       'claim': 'Автор разрешил остановку и отдельный вариант; критерии технического совпадения записаны до замера'}],
         'cannot_claim': 'Ускорение выполнения не является улучшением HOTA. Новый предел HOTA и независимая проверка не разрешены.'})
    catpath.write_text(json.dumps(cat, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    plan = ROOT / 'dissertation/ЕДИНЫЙ_ПЛАН.md'
    text = plan.read_text(encoding='utf-8')
    entry = f'''**{now:%d.%m.%Y, %H:%M} MSK** — автор остановил HOTA для отдельной пробы Triton

*Постановка (`z13_triton_execution_v1`).* Автор попросил сейчас остановить обычный расчёт и запустить отдельный вариант Triton. `runs/z13_spatial_hota_v1_20261014/status.json`: paused, 6 завершённых заданий; готовые видео сохранены, прерванное не принято. Автоматизация приостановлена во избежание дублирования. Зависимость Triton 3,6 устанавливается отдельно от действующего окружения. Новый адаптер компилирует тензорный шаг B и шаг нейронной динамики C/D; веса, вход, считыватель, пороги и окружение трекера сохраняются. `runs/z13_triton_hota_v1_20261014/protocol.json`: до измерения записаны допуски состояния/вероятностей, совпадение решений и парное время видео. Срок до {auth['deadline']} не продлевается. Полный запуск только при техническом совпадении и выигрыше времени; обучения и отложенных данных нет. HOTA и скорость нового исполнения ещё не измерены.

'''
    assert '## Журнал изменений\n\n' in text
    plan.write_text(text.replace('## Журнал изменений\n\n', '## Журнал изменений\n\n' + entry, 1), encoding='utf-8')
    print(json.dumps(spec, ensure_ascii=False))


if __name__ == '__main__':
    main()
