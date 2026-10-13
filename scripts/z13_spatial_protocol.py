"""Freeze the authorized v2 training protocol before the first GPU step."""
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tracking.z13_checkpoint import atomic_json
from scripts.z13_train import split_videos

RUN = ROOT / 'runs/z13_spatial_v2_20261013'
ID = 'fly_spatial_quality_z13_v2'


def main():
    fit, tau = split_videos()
    import numpy as np
    perm = np.random.default_rng(13).permutation(fit).tolist()
    halves = [sorted(perm[:52]), sorted(perm[52:])]
    protocol = {
        'id': ID, 'authorization': 'User 2026-10-13: Запускай что надоо для обучения',
        'scope': 'Technical probe, B/C/D localization training, two-fold train-only predictions, quality training and threshold selection. No dev/MASA/heldout evaluation or downloads.',
        'fit': fit, 'tau': tau, 'halves': halves,
        'arms': ['B', 'C', 'D'], 'seed': 0, 'steps': 4000, 'batch': 16, 'micro_batch': 4,
        'readout_width': 32, 'template_points': 64,
        'readout_lr': .003, 'anchor': 1.0,
        'quality_steps': 1500, 'quality_batch': 256, 'quality_lr': .003,
        'quality_features': 12, 'quality_hidden': 16,
        'threshold_precision': .9, 'minimum_interventions': 50, 'success_scale': .25,
        'selection': 'Exact unique scores, ties kept; minimum threshold satisfying original rule. Null => no intervention. Natural episodes only for quality fitting and threshold.',
        'technical_steps_per_arm': 32,
        'technical_gates': ['finite losses/gradients', 'nonzero object gradient into D core',
                            'template change changes spatial logits', 'checkpoint optimizer and RNG exact restore',
                            'geometry round-trip and labels in valid columns', 'all arms retain identical readout size'],
        'safety': {'one_gpu_worker': True, 'priority': 'Idle', 'save_every_steps': 10,
                   'save_seconds': 60, 'stall_seconds': 600, 'maximum_stall_retries': 2},
        'jobs': [{'arm': arm, 'name': name, 'fit': f, 'predict': h}
                 for arm in ['B', 'C', 'D']
                 for name, f, h in [('fold0', halves[0], halves[1]), ('fold1', halves[1], halves[0]), ('full', fit, tau)]],
        'limitations': ['One seed; train known to detector; dev repeatedly used in development',
                       'OOF quality training uses 52-video localizers; final localizer uses 104, so quality distribution shift remains possible',
                       'Two readout changes; presence-only evaluation is a diagnostic ablation, not causal proof about all architecture components'],
    }
    RUN.mkdir(exist_ok=True)
    p = RUN / 'protocol.json'
    if p.exists():
        if json.loads(p.read_text(encoding='utf-8')) != protocol:
            raise ValueError('Refusing to replace frozen protocol')
    else:
        atomic_json(p, protocol)
    catalog_path = ROOT / 'experiments/project_history.json'
    catalog = json.loads(catalog_path.read_text(encoding='utf-8'))
    if any(c['id'] == ID for c in catalog['experiments']):
        print('Protocol already recorded'); return
    parent = next(c for c in catalog['experiments'] if c['id'] == 'fly_object_readout_z13_v1')
    catalog['experiments'].append({
        'id': ID, 'title': 'З13 v2: пространственное считывание FlyVis и качество положения',
        'area': parent['area'], 'state': 'planned', 'started': '2026-10-13',
        'question': 'Исправляет ли пространственное считывание и оценка качества функциональный отказ З13, и есть ли добавка FlyVis?',
        'hypothesis': 'Сохранение пространственных признаков образца и обучение уверенности полезной поправке могут помочь; первопричина прежнего провала не установлена.',
        'expected': 'Автор разрешил подготовку и обучение. Одна версия: B без FlyVis, C замороженная, D обучаемая; одинаковое считывание. По 4000 шагов на всех 104 train и на двух половинах по 52 для прогнозов вне обучения; оценщик качества обучается только на этих прогнозах. Прежние 26 train — только τ. Протокол runs/z13_spatial_v2_20261013/protocol.json. Никаких dev/MASA/75 val в этом запуске.',
        'acceptance': [
            {'criterion': 'Все заранее записанные технические проверки пройдены; сохранение и возобновление проверены', 'met': None},
            {'criterion': 'На 26 train у D выбран точный порог с минимум 50 вмешательствами и минимум 90% успехов; успех — прежний предел 0,25 размера и улучшение против прогноза', 'met': None},
            {'criterion': 'Польза стенду: после отдельного согласования оценки D−стенд с MASA не меньше +0,10 общей HOTA и интервал парных разностей по видео выше нуля', 'met': None},
            {'criterion': 'Добавка FlyVis: после отдельной оценки интервал D−B по HOTA выше нуля; польза обучения ядра отдельно требует интервала D−C выше нуля', 'met': None},
        ],
        'cannot_claim': 'Предложение и технические проверки не доказывают прирост HOTA или причину прежнего отказа. Провал этой версии не закрывает FlyVis. Правило 90%/50 инженерное, не гарантия пользы ассоциации.',
        'evidence': [{'kind': 'json', 'ref': 'runs/z13_spatial_v2_20261013/protocol.json', 'claim': 'Зафиксирован до первого GPU-шага'},
                     {'kind': 'file', 'ref': 'research_z13_spatial_readout_2026-10-13/СВОДКА.md', 'claim': 'Литературная опора и критика'}],
    })
    catalog_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    plan = ROOT / 'dissertation/ЕДИНЫЙ_ПЛАН.md'
    content = plan.read_text(encoding='utf-8')
    now = datetime.now().astimezone()
    entry = f'''**{now:%d.%m.%Y, %H:%M} MSK** — З13 v2: автор разрешил обучение пространственного считывания

*Постановка (`{ID}`).* После литературного обзора автор разрешил «Запускай что надоо для обучения». Зафиксирован `runs/z13_spatial_v2_20261013/protocol.json`: один пространственный считыватель ширины 32, до 64 колонок образца, отдельный оценщик качества по 12 наблюдаемым характеристикам карты и геометрии. B/C/D — одинаковые действия и считывание; данные, разметка, ядро и его прежние настройки не заменяются. По 4000 шагов на полном train 104 и двух половинах по 52 для прогнозов вне обучения. Качество учится на этих прогнозах, прежние 26 train служат только для точного выбора τ; неизвестная идентичность не становится отрицательной меткой. Допуск прежний: минимум 50 вмешательств и 90% успехов. Техническая проба не выбирает архитектуру по качеству на dev.

*Решение.* Один процесс GPU с низким приоритетом, пакет 16 накоплением четырёх частей по 4, сохранения оптимизатора/генераторов/версии данных и кода. Контроллер ограничивает повторы зависших собственных заданий; чужие процессы не останавливает. Новый запуск относится к обучению и выбору τ; dev/MASA и отложенные 75 val не запускаются. Срок уточняется по пробе, GPU занят игрой. Это постановка, HOTA не измерялась.

'''
    plan.write_text(content.replace('## Журнал изменений\n\n', '## Журнал изменений\n\n' + entry, 1), encoding='utf-8')
    print(ID, 'recorded; no GPU started')


if __name__ == '__main__':
    main()
