"""Audit the completed native-only benchmark; never resume the pending FlyVis gate."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from scripts.z13_candidate_reserve_20261021 import read, write, sha, now

W = ROOT / 'runs/z13_weekend_20261024'
RUN = ROOT / 'runs/z13_val20_matched_20261024'
STAGE = RUN / 'native_masa_recovery2'


def main():
    import numpy as np
    from scripts.z13_hota_deadline_guard import WindowsProcesses
    from scripts.z13_localization_labels_20261023 import verify_sources
    from scripts.z13_continue import paired

    protocol = read(STAGE / 'protocol.json')
    verify_sources(protocol)
    result = read(STAGE / 'controller_result.json')
    assert result['exit_code'] == 0 and result['worker_ended']
    assert result['all_owned_ended'] and not result['remaining']
    identities = dict(result['owned'])
    for name in ('controller.json', 'completion.json'):
        item = read(STAGE / name)
        identities[str(item['pid'])] = item['birth']
    ps = WindowsProcesses()
    remaining = {p: b for p, b in identities.items() if ps.creation(int(p)) == b}
    assert not remaining, remaining
    report = read(STAGE / 'report.json')
    assert report['videos'] == protocol['videos'] and len(report['videos']) == 20
    assert report['all_conditions_common'] and not report['independent_final_test']
    jobs = list((STAGE / 'jobs').glob('*.json'))
    assert len(jobs) == 105
    checked = {}
    for marker in jobs:
        for name, digest in read(marker)['outputs'].items():
            path = (ROOT / name).resolve()
            path.relative_to(ROOT)
            assert sha(path) == digest, name
            checked[name] = digest
    assert len(report['output_sha256']) == 160
    for name, digest in report['output_sha256'].items():
        assert sha(ROOT / name) == digest, name
    assert len(report['pooled']) == len(report['per_video']) == 8
    for arm in ('xywh', 'bytetrack', 'botsort', 'tracktrack'):
        for video in protocol['videos']:
            with np.load(RUN / arm / 'signatures' / video / 'signatures.npz') as a:
                with np.load(RUN / 'raw' / arm / 'raw' / (video + '.npz')) as b:
                    assert np.array_equal(a['outputs'], b['outputs']), (arm, video)
    pv = {a: {v: x['HOTA'] for v, x in rows.items()}
          for a, rows in report['per_video'].items()}
    for name, expected in report['paired'].items():
        a, b = name.split('-')
        assert paired(pv[a], pv[b]) == expected and expected['n'] == 20
    digest = sha(STAGE / 'report.json')
    assert read(STAGE / 'finish_recorded.json')['report_sha256'] == digest
    assert read(STAGE / 'verified_summary.json')['report_sha256'] == digest
    assert read(W / 'val20_native_masa_recovery2_verified.json')['checks_green']
    assert '58 passed' in (W / 'logs/val20_native_masa_recovery2_tests.stdout.log').read_text(encoding='utf-8')
    protected = read(W / 'vault_preparation.json')['protected_existing_edits']
    for name, expected in protected.items():
        actual = sha(ROOT / name) if (ROOT / name).is_file() else None
        assert actual == expected, ('Foreign edit changed', name)
    pending = read(W / 'val20_replay_pending_author_decision.json')
    assert pending['status'] == 'pending'
    baseline = report['pooled']['xywh_masa']['HOTA']
    comparisons = {}
    for arm in ('xywh', 'bytetrack', 'botsort', 'tracktrack'):
        raw = report['pooled'][arm + '_raw']['HOTA']
        masa = report['pooled'][arm + '_masa']['HOTA']
        comparisons[arm] = dict(raw=raw, masa=masa,
            masa_relative_percent_from_xywh=100 * (masa-baseline) / baseline,
            masa_relative_percent_from_own_raw=100 * (masa-raw) / raw)
    write(W / 'native20_final_audit.json', dict(at=now().isoformat(),
        report_sha256=digest, verified_jobs=len(jobs),
        verified_job_outputs=len(checked), verified_report_outputs=160,
        raw_outputs_exactly_equal_independent_cpu=80,
        paired_intervals_recomputed=len(report['paired']),
        original_monitor_recomputed_pooled_hota=8,
        checked_process_identities=len(identities), all_identities_ended=True,
        protected_foreign_edits=len(protected), gitignore_sha256=sha(ROOT / '.gitignore'),
        comparison=comparisons, no_flyvis=True,
        next_required_action='Explicit author decision on pending replay tolerance; no FlyVis job resumed.',
        scientific_tolerances_changed=False))
    from scripts.z13_val20_native_masa_20261024 import journal
    journal('независимая сверка полного базиса20val перед публикацией',
        '*Получено.* runs/z13_weekend_20261024/native20_final_audit.json: '
        'проверены105маркеров,160выходов отчёта,80точных raw-сверок с независимым CPU;'
        '10парных интервалов воспроизведены. Монитор ранее повторно проверил8общих HOTA;'
        'контроллер exit0,все зарегистрированные PID/времена создания и монитор окончены. '
        'Источник чисел runs/z13_val20_matched_20261024/native_masa_recovery2/report.json, '
        'SHA256='+digest+'.\n\n'
        '*Вывод.* TrackTrack с общей MASA59,62236652 против XYWH58,48578688;'
        'средняя разность по20видео+1,17631053[0,77274756;1,60453322]. '
        'Самой XYWH сшивка дала среднюю+0,69966882[0,35687953;1,07055797]. '
        'Это базис окружения,не улучшение FlyVis;20val не независимый итоговый тест.\n\n'
        '*Решение.* Результат сохранить и опубликовать точными файлами. '
        'Ответа на отдельный вопрос о служебном допуске пока нет;'
        'FlyVis-ветвь не возобновляется,новый heartbeat согласия не даёт. '
        'Другой готовой необходимой стадии без этого решения нет;'
        'не запускать повторные базисные расчёты ради занятости. '
        'Проваленные descriptor/loss-версии к HOTA не продвигаются. '
        'Архитектурного изменения нет. 57чужих графправок и .gitignore сохранены. '
        'TrackTrack+MASA к XYWH+MASA:+1,94334334%;доля измеренного потолка не определена.')
    handoff = W / 'HANDOFF.md'
    text = ('# Актуально — '+now().strftime('%d.%m.%Y %H:%M')+' MSK\n\n'
        'native_masa_recovery2 ПОЛНОСТЬЮ ЗАВЕРШЁН11:25:48,exit0/all_owned_ended. '
        'Монитор записал итог11:26:54,проверки каталога/58тестов/граф прошли11:28:15. '
        'Отдельный аудит native20_final_audit.json проверяет105jobs,160выходов отчёта,'
        '80точных raw-сверок,10парных интервалов и фактический конец всех PID/birth. '
        'Повторно исходные видео/GPU не считались.\n\n'
        'Один20val-базис,raw / общаяMASA:XYWH57,90082918 /58,48578688;'
        'Byte55,72093730 /56,30107748;BoT58,49031414 /58,87367976;'
        'Track59,26475008 /59,62236652. '
        'TrackMASA-XYWHMASA средняяповидео+1,17631053[0,77274756;1,60453322],17/20. '
        'Источник native_masa_recovery2/report.json,verified_summary.json,results.md. '
        'Не смешивать общую HOTA со средним повидео;это не результат FlyVis '
        'и не нетронутый финальный тест. Карточка z13_val20_native_masa_v1 завершена;'
        'монитор сохранил прежние итоговые поля в revisions.\n\n'
        'ОЖИДАЕТСЯ ЯВНОЕ РЕШЕНИЕ АВТОРА:val20_replay_pending_author_decision.json '
        'status=pending. Ответа на уже заданный вопрос о допуске0,0001px нет. '
        'Не менять допуск и не возобновлять FlyVis по heartbeat/тишине;'
        'не задавать тот же вопрос повторно. common остаётся inconclusive,старый '
        'точный контроль не пройден. Нет готовой разрешённой следующей стадии '
        'до этого решения;не повторять базис,не продвигать отрицательные версии '
        'descriptor/loss к HOTA. GPU-расчёты закончены;окно до26.10 07:40МСК '
        'само по себе не снимает этот запрет. Автоматизацию не учащать.\n\n'
        'Публикация и проверки этого аудита:native20_final_checks.json и '
        'native20_final_publication.json. Чужие57графправок/.gitignore сохранены,'
        'архитектурного изменения нет. Старые коды ошибок и все стадии сохранены.\n\n'
        'Ниже прежние передачи сохранены.\n\n')
    handoff.write_text(text + handoff.read_text(encoding='utf-8'), encoding='utf-8')
    print('Audit verified:105jobs,160report outputs,80raw replays,10paired intervals;all own identities ended.', flush=True)


if __name__ == '__main__':
    main()
