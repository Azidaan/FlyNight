"""Freeze one new diagnostic HOTA policy, preserving the failed v2 gate."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCE = ROOT / 'runs/z13_spatial_v2_20261013'
RUN = ROOT / 'runs/z13_spatial_hota_v1_20261014'
ID = 'fly_spatial_hota_policy_z13_v1'


def main():
    import torch
    from tracking.z13_checkpoint import atomic_json
    from tracking.z13_spatial_localizer import percentile_threshold
    if (RUN / 'protocol.json').exists():
        raise ValueError('Frozen protocol exists; do not rewrite')
    RUN.mkdir(parents=True, exist_ok=True)
    source_protocol = json.loads((SOURCE / 'protocol.json').read_text(encoding='utf-8'))
    dev = json.loads((ROOT / 'runs/train_cache/selection.json').read_text(encoding='utf-8'))['videos']['dev']
    assert len(dev) == 35
    protocol = {'id': ID, 'created': datetime.now().astimezone().isoformat(),
        'authorization': 'Автор: давай, к предложению подготовить один опыт HOTA',
        'scope': 'Preparation and bounded technical probe; full GPU run budget to be agreed after measured price. No training, no heldout videos, no downloads.',
        'source': str(SOURCE.relative_to(ROOT)), 'videos': sorted(dev), 'arms': ['B', 'C', 'D'],
        'policy': 'Each arm: threshold at top 1% of all 8326 natural train26 predictions, all ties retained. No kind/error/HOTA used in selection. Same fixed percentile; no hard intervention cap in closed loop.',
        'fraction': .01, 'threshold_videos': source_protocol['tau'],
        'thresholds': {}, 'model_sha256': {},
        'action': 'Lost tracks only, existing 30-frame buffer: score>=tau shifts predicted center to spatial readout position. Velocity, size, covariance and association rules unchanged.',
        'pipeline': 'YOLO cached detections, XYWHByteTrack 30Hz, same lamina eye input, same base+MASA stitching gap150 dist5 tau0.6.',
        'comparisons': ['D minus stand', 'D minus B', 'D minus C (separate core-training claim)'],
        'primary_metric': 'Paired mean HOTA difference per video with 95% bootstrap CI; pooled HOTA reported separately.',
        'acceptance': {'D_vs_stand': 'pooled gain>=0.10 and paired CI lower>0',
                       'D_vs_B': 'paired CI lower>0', 'D_vs_C': 'paired CI lower>0'},
        'technical': 'No-intervention output exactly matches baseline; stored-episode spatial probabilities agree; only lost tracks and centers changed; bounded batches of four; all arms load.',
        'safety': {'one_gpu_process': True, 'idle_priority': True, 'max_batch': 4,
                   'no_auto_retuning': True, 'probe_timeout_seconds': 900},
        'limitations': ['Dev repeatedly used; train26 threshold data already used in development',
                       'This is a new diagnostic policy; failed 90%/50 gate stays unchanged',
                       'Score quantile selected on annotated-time natural examples; runtime acts on all 30Hz lost steps and may have a different distribution',
                       'No safety claim from this percentile or from a positive HOTA sign'],
    }
    for arm in protocol['arms']:
        rows = json.loads((SOURCE / arm / 'threshold_predictions.json').read_text(encoding='utf-8'))
        assert len(rows) == 8326
        selection = percentile_threshold([r['p_present'] for r in rows], .01)
        source = SOURCE / arm / 'model.pt'
        model = torch.load(source, map_location='cpu', weights_only=False)
        assert model['format'] == 'z13-spatial-v2' and model['tau']['tau'] is None
        protocol['thresholds'][arm] = selection
        protocol['model_sha256'][arm] = hashlib.sha256(source.read_bytes()).hexdigest()
        folder = RUN / arm
        folder.mkdir(exist_ok=True)
        model.update(format='z13-spatial-hota-policy-v1', tau={'tau': selection['tau']},
                     policy={'fraction': .01, 'source_sha256': protocol['model_sha256'][arm]})
        temporary = folder / 'model.pt.next'
        torch.save(model, temporary)
        check = torch.load(temporary, map_location='cpu', weights_only=False)
        assert check['tau']['tau'] == selection['tau']
        temporary.replace(folder / 'model.pt')
        atomic_json(folder / 'policy.json', selection)
    atomic_json(RUN / 'protocol.json', protocol)
    atomic_json(RUN / 'videos.json', {'videos': protocol['videos']})
    atomic_json(RUN / 'probe_video.json', {'videos': protocol['videos'][:1]})
    catalog_path = ROOT / 'experiments/project_history.json'
    catalog = json.loads(catalog_path.read_text(encoding='utf-8'))
    assert not any(c['id'] == ID for c in catalog['experiments'])
    catalog['experiments'].append({'id': ID, 'title': 'З13 v2: HOTA при одной политике верхнего 1% уверенности',
        'area': 'flyvis', 'state': 'planned', 'started': '2026-10-14',
        'question': 'Даёт ли готовый пространственный сигнал FlyVis пользу HOTA при одной новой политике вмешательства?',
        'hypothesis': 'Провал инженерного 90%/50 допуска не определяет HOTA; одна фиксированная политика позволит проверить фактический эффект.',
        'expected': 'Одна рабочая точка B/C/D: верхний 1% всех естественных предсказаний на train26, без меток и HOTA в выборе; все равные оценки вместе. Полный трекер на dev35 с прежней MASA, стенд и B/C/D, без нового обучения. Это новый опыт, старый итог fly_spatial_quality_z13_v2 сохраняется. Подготовка и короткая проба разрешены; полный бюджет после замера.',
        'acceptance': [{'criterion': protocol['technical'], 'met': None},
                       {'criterion': 'D−стенд: общая HOTA минимум +0,10 и нижняя граница парного ДИ выше нуля', 'met': None},
                       {'criterion': 'Вклад FlyVis: нижняя граница парного ДИ D−B выше нуля', 'met': None},
                       {'criterion': 'Польза обучения ядра отдельно: нижняя граница парного ДИ D−C выше нуля', 'met': None}],
        'cannot_claim': 'Проба не подтверждает HOTA. Верхний 1% не гарантирует безопасность или такую же частоту вмешательств на dev. Dev — разработка; независимые видео не открываются. Исходный отрицательный критерий не переписывается.',
        'evidence': [{'kind': 'json', 'ref': 'runs/z13_spatial_hota_v1_20261014/protocol.json', 'claim': 'Политика и критерии до HOTA'},
                     {'kind': 'json', 'ref': 'runs/z13_spatial_v2_20261013/final_verification.json', 'claim': 'Исходное завершённое обучение и провал прежнего допуска'}]})
    catalog_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    now = datetime.now().astimezone()
    entry = f'''**{now:%d.%m.%Y, %H:%M} MSK** — З13 v2: подготовка одной новой политики для HOTA

*Постановка (`{ID}`).* Автор согласился подготовить один опыт HOTA с готовыми весами. До HOTA зафиксирован `runs/z13_spatial_hota_v1_20261014/protocol.json`: B/C/D выбирают порог по верхнему 1% всех 8326 естественных предсказаний на прежних 26 train, без kind, истинной ошибки и HOTA в выборе; равные оценки не разделяются. Метки не определяют включение в выборку, неизвестные включены в распределение оценок. Частота на dev не ограничена этим процентом и может отличаться. Готовые веса не обучаются. В прежнем опыте τ=null и отрицательный итог сохранены.

*Действие и критерии.* Только потерянный трек в прежнем буфере 30 кадров: при уверенности ≥ τ заменить центр прогноза найденной точкой. Полный выход с прежней MASA; D−стенд общая HOTA ≥ +0,10 и парный ДИ по видео выше нуля, вклад FlyVis D−B и обучение ядра D−C — отдельно, каждый с ДИ выше нуля. Dev35 — разработка; отложенные данные не открываются. Один порог на вариант до просмотра HOTA, повторного подбора нет.

*Решение.* Подготовить подключение, сверить отсутствие вмешательств со стендом, соответствие считывания сохранённым эпизодам и цену по одной короткой пробе. Полный GPU-прогон после измерения цены и согласования бюджета. Подготовка не является измерением HOTA; процент прироста неизвестен.

'''
    plan = ROOT / 'dissertation/ЕДИНЫЙ_ПЛАН.md'
    text = plan.read_text(encoding='utf-8')
    plan.write_text(text.replace('## Журнал изменений\n\n', '## Журнал изменений\n\n' + entry, 1), encoding='utf-8')
    print(json.dumps({'id': ID, 'thresholds': protocol['thresholds'], 'probe_video': protocol['videos'][0]}))


if __name__ == '__main__':
    main()
