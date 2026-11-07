"""Регистрация новой ограниченной проверки Б2, без продления матрицы истории."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.continuation_service_20261028 import RUN, service


def main():
    folder = RUN / 'b2_causal960_profile'
    assert not folder.exists()
    proof = service.read(RUN / 'b2_manifest_checks_recovery1/report.json')
    assert proof['passed']
    assert proof['source_controller_sha256'] == service.sha(RUN / 'b2_causal_manifest/controller_result.json')
    manifests = service.read(RUN / 'b2_causal_manifest/report.json')
    entry = next(e for e in manifests['entries'] if e['split'] == 'fit84')
    assert service.sha(ROOT / entry['manifest']) == entry['sha256']
    video = entry['video']; manifest = service.read(ROOT / entry['manifest'])
    files = ['tracking/b2_causal_detector.py', 'tracking/b2_feature_temporal.py',
             'scripts/profile_b2_causal960_20261028.py', 'tests/test_b2_causal_detector.py',
             'weights/baseline_seed42/best.pt', entry['manifest'], manifest['alignment'],
             manifest['gt_only_detection_targets'], f'runs/train_cache/{video}/detections/raw_video.json',
             f'runs/train_cache/{video}/detections/raw_video.npz',
             'runs/continuation_20261028/b2_manifest_checks_recovery1/report.json',
             'runs/continuation_20261028/b2_causal_manifest/controller_result.json']
    protocol = dict(id='b2_causal960_path_v1', created=service.now(), budget_seconds=900,
        timezone='Europe/Moscow', save_at='2026-10-29T07:30:00+03:00',
        manifest=entry['manifest'], cache=f'runs/train_cache/{video}/detections/raw_video.npz',
        action='Frozen fused YOLO26s P3, same960/conf.01/iou.7 as retained cache; ring of five real query frames. '
               'Common additive .1 adapter to first detection head input; no new edges/weights/data/environment.',
        controls='Ordinary same HexGRU/initialization and identical adapters for five real features, repeated last, repeated arithmetic mean; '
                 'separately trainable in future, no optimizer updates here. Existing C/D/R1 remain untrained scientific branches.',
        timestamps='Strict saved raw frame times/FPS; FlyVis future integration uses actual interval and step<=.02; first1/30 interval is not .2.',
        targets='GT only last-query targets; exact LetterBox integer padding/clipping. crowd class anchors do not receive background BCE; '
                'positive assigned targets override ignore; bbox/DFL losses remain native. Policy fixed before training.',
        acceptance=['Full first video query traversal and exact zero raw/cache detections at960',
                    'Bounded ring with exact manifest five-frame chronology and no future inputs',
                    'Real one-to-one loss backward for history/last/mean finite,YOLO parameters unchanged,no optimizer steps'],
        strongest_counterargument='Fixed P3 location does not follow a moving object; this path does not implement motion alignment.',
        limitations='Infrastructure only. One historical train video, no detector accuracy/HOTA/no scientific verdict/no trained promoted weights. '
                    'Strong last/mean share ordinary trainable architecture but are not pure order-only controls. No matrix price override.',
        sha256={f:service.sha(ROOT/f) for f in files})
    service.write(folder / 'protocol.json', protocol)
    history = service.read(ROOT / 'experiments/project_history.json')
    base = next(c for c in history['experiments'] if c['id'] == 'b2_causal_manifest_v1')
    assert not any(c['id'] == protocol['id'] for c in history['experiments'])
    history['experiments'].append(dict(id=protocol['id'], title='Б2 — причинный путь детектора на 960 и контроль последнего кадра',
        area=base['area'], state='ready', started='2026-10-28',
        question='Сохраняет ли новый реальный путь Б2 исходные обнаружения при нулевой поправке на всём одном видео?',
        hypothesis='Ограниченное кольцо и общие адаптеры позволяют проверить действие до обучения без повторной выгрузки P3.',
        expected='Точный native/zero-cache, причинные пять кадров, конечная производная истинной функции ошибки без изменения YOLO.',
        acceptance=[dict(criterion=s, met=None) for s in protocol['acceptance']],
        evidence=['runs/continuation_20261028/b2_causal960_profile/protocol.json'],
        cannot_claim=protocol['limitations']))
    service.write(ROOT / 'experiments/project_history.json', history)
    from scripts.b123_safe_journal_20261027 import journal
    journal(protocol['id'], 'Зарегистрирован полный причинный проход Б2 на 960 до нового обучения',
            'Постановка и бюджет900сGPU: runs/continuation_20261028/b2_causal960_profile/protocol.json. '
            'Проверяется полный один train-видеопроход, точный cache/native/zero, кольцо5, реальные интервалы, '
            'отдельный crowd-ignore/last/mean, без optimizer.step/новых обученных весов. '
            'Это не продление остановленной матрицы истории и не повтор старых трёх scratch-шагов. '
            'Перед GPU обязательны каталог и62CPUтеста; во время них план/каталог/контролируемые отчёты не изменять.')
    for name, budget, script in [('b2_causal960_checks', 600, 'scripts/check_b2_causal960_20261028.py'),
                                 ('b2_causal960_record', 600, 'scripts/record_b2_causal960_20261028.py')]:
        service.write(RUN / name / 'protocol.json', dict(budget_seconds=budget, created=service.now(),
                      sha256={script:service.sha(ROOT / script)}))
    script = 'scripts/check_b2_causal960_20261028.py'
    service.write(RUN / 'ready_jobs/112_b2_causal960_checks.json', dict(name='b2_causal960_checks',
                  folder=(RUN / 'b2_causal960_checks').relative_to(ROOT).as_posix(), script=script,
                  script_sha256=service.sha(ROOT / script), protocol_sha256=service.sha(RUN / 'b2_causal960_checks/protocol.json')))


if __name__ == '__main__': main()
