"""Регистрация полного Б1 до CUDA; публикация только после проверок каталога."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.recovery_research_service_20261027 import RUN, read, write, sha, now
IDENT = 'b1_full_tracker_pilot_v1'


def prepare():
    folder = RUN / 'b1_tracker'
    assert not folder.exists()
    auth = read(RUN / 'authorization.json')
    previous = read(RUN / 'lazy_appearance_validation/protocol.json')
    sources = [s for s in previous['sources'] if s['group'] == 'development20']
    assert len(sources) == 20
    strong = 'runs/z13_val20_matched_20261024/native_masa_recovery2/report.json'
    report = read(ROOT / strong)
    assert set(report['per_video']['xywh_masa']) == {s['video'] for s in sources}
    trained = read(ROOT / 'runs/b123_20261027/b1_training/report.json')
    assert trained['complete'] and not trained['hota_measured']
    files = ['tracking/b1_tracker_runtime.py', 'scripts/run_b1_tracker_20261028.py',
        'scripts/audit_b1_tracker_20261028.py', 'scripts/prepare_b1_tracker_20261028.py',
        'tests/test_b1_tracker_runtime.py', 'tracking/b1_appearance.py', 'tracking/b1_appearance_fast.py',
        'tracking/b1_learning.py', 'tracking/b1_real_inputs.py', 'tracking/flyvis_temporal_residual.py',
        'tracking/b1_fast_stimulus.py', 'tracking/b1_index_copy_stimulus.py',
        'tracking/z13_readout.py', 'tracking/z13_region_assignment.py',
        'tracking/z13_region_assignment_context.py', 'tracking/foveated_eyes.py', 'tracking/retina.py',
        'tracking/masa_r50.py', 'tracking/xywh_bytetrack.py', 'tracking/offline_association.py',
        'scripts/run_bdd_video_rate_experiment.py', 'scripts/z13_masa_exemplar_snap_20261026.py',
        'weights/masa/masa_r50.pth', 'runs/b1_real_inputs_20261026/identity_fold4.joblib',
        'runs/b1_real_inputs_20261026/gate.joblib',
        'runs/b123_20261027/b1_base_export_vectorized/recovery1/outputs/pca.npz',
        'runs/b123_20261027/b1_base_export_vectorized/recovery1/outputs/report.json',
        'runs/b123_20261027/b1_training/protocol.json', 'runs/b123_20261027/b1_training/report.json',
        'runs/b123_20261027/b1_residual_bound/report.json', 'runs/b1_real_inputs_20261026/gpu_samples.json',
        'runs/b1_real_inputs_20261026/base_predictions.npz', strong]
    files += [f'runs/b123_20261027/b1_training/{a}/checkpoint.pt' for a in ['C', 'D', 'R1', 'B']]
    for s in sources:
        files += [s[k] for k in ['baseline', 'detections', 'alignment', 'raw_array']]
    p = dict(parent_experiment=IDENT, created=now(), budget_seconds=10800, audit_budget_seconds=1800,
        author_save_at=auth['save_at'], author_stop=auth['controller_stop'], config=previous['config'],
        sources=sources, strong_reference=strong, variants=['baseline', 'appearance', 'zero', 'C', 'D', 'R1', 'B'],
        action='Настоящее первое назначение XYWHByteTrack: сохранённая identity-модель MASA+геометрия+память, затем ограниченная поправка старой обученной головы, прежний gate и порог0.5. Выбранное действие применяется categorical_cost ДО LAP; следующие назначения/состояния развиваются заново. Остальные стадии исходные.',
        history='Причинная lamina/4, 721колонка, четыре последних реально ведомых наблюдения с повтором первого при нехватке. Образец слот3, его наблюдаемая маска; дальнейший вход на каждом кадре потери. Никакие GT-ID/расписания аннотаций не выбирают входы. Предыдущее решение памяти — исходное назначение до поправки.',
        appearance='Прежний MASA-R50 FP32, полный кадр длинной стороной1024, тот же L2 и квантование выходных векторов float16→float32 старого банка. Масочный образец+snap к текущей наблюдаемой детекции в прошлом кадре при IoU>=.3. Никаких новых весов/повторного экспорта28970.',
        fixed='Все четыре checkpoint одной уже завершённой эпохи/seed26; identity_fold4, gate, PCA и порог0.5 заморожены. Никакого дообучения/выбора по20val. ШтатнаяC, обученнаяD, входглазаR1 и обычнаяHexGRU B имеют прежние головы; B не согласована по числу параметров. zero реально прогоняетD, но обнуляет delta.',
        cost_gate='Первое ПОЛНОЕ видео7вариантов и метрики:2*time*19+300<остаток10800с. Иначе сохранить это видео/цену, остальные19не продолжать, бюджет не продлевать, научного вывода по20нет. До первого видео одна фиксированная медианная история проверяет точное равенство нового инкрементального пути прежнему вычислению.',
        technical='Все20 исходных30Гц raw-массивов/JSON точно совпадают; zero и appearance совпадают на каждом кадре по выходу и Kalman mean/cov; веса/пороги неизменны; CPU-повтор всех назначений/сырыхвыходов/голов/метрик; независимые парные интервалы, конец деревьев.',
        primary='Это пилот одной уже обученной реализации. ПользаD: положительная нижняя95%граница парной HOTA к baseline, той же appearance, C, R1 и B. Среднюю/совокупную HOTA не смешивать. Отдельно сопоставить со старой сильной офлайнMASA на тех же20, но её временной режим другой. Полное научное подтверждение Б1 требует также воспроизводимости по сидам, этот пилот его не заменяет.',
        counterargument='Одна эпоха не подтвердила saved-query пользу; ±1логит не переворачивает разрыв>2. Просто продлевать старое обучение нельзя. Старый gate использовал последствия full-box в своей разметке, а сохранённый Б1 helper задаёт categorical_cost: saved-query utility НЕ равна эффекту этого живого действия. У всех сравниваемых Б1-путей действие одинаковое; реальный исход измерит только HOTA. Живые запросы/память теперь каждый кадр, тогда как обучение выбирало сохранённые запросы: возможен сдвиг распределения.',
        limitation='20 исторических val, уже многократно использованных, не новый независимый тест. Реальные рамки класса в поле глаза, categorical_cost может разрешить ребро за прежним геометрическим порогом; это явно выбранное действие текущего Б1 helper, не обещание неизменной маски ByteTrack. Прирост общей системы и дополнительный вклад ядра отдельно. При ценовом/техническом стопе Б1 остаётся непроверенной по полной HOTA.',
        literature='Существующая постановка: dissertation/Новые гипотезы — временная ветвь и память объектов.md, раздел2; FlyVis Nature2024 DOI10.1038/s41586-024-07939-3 и MOTR ECCV2022. Новая теория не выдвигается: интегрируются сохранённые обученные веса.',
        sha256={f: sha(ROOT / f) for f in sorted(set(files))})
    write(folder / 'protocol.json', p)
    write(RUN / 'b1_tracker_audit/protocol.json', dict(parent_experiment=IDENT, budget_seconds=1800, source='b1_tracker'))
    catalog = read(ROOT / 'experiments/project_history.json')
    assert not any(c['id'] == IDENT for c in catalog['experiments'])
    catalog['experiments'].append(dict(id=IDENT, title='Б1 — Временная поправка к внешнему виду внутри полного трекера', area='flyvis',
        state='ready', started='2026-10-28', question='Каков полный эффект уже обученной временной поправки Б1 при реальных последовательных назначениях?',
        hypothesis='Рабочая гипотеза Б1: D имеет дополнительный полезный сигнал относительно той же базы, C, R1 и B; результат заранее неизвестен.',
        expected=p['action'], acceptance=[dict(criterion=p['technical'], met=None), dict(criterion=p['primary'], met=None)],
        obtained='', interpretation='', decision='', cannot_claim=p['limitation'] + ' ' + p['counterargument'],
        evidence=[dict(kind='file', ref=(folder / 'protocol.json').relative_to(ROOT).as_posix(), claim='До CUDA: фиксированные веса, действие, контроли, критерии и стоимость')],
        files=['tracking/b1_tracker_runtime.py', 'scripts/run_b1_tracker_20261028.py', 'scripts/audit_b1_tracker_20261028.py']))
    main = next(c for c in catalog['experiments'] if c['id'] == 'flyvis_temporal_residual_association_v1')
    main['evidence'].append(dict(kind='file', ref=(folder / 'protocol.json').relative_to(ROOT).as_posix(), claim='Новая реальная интеграция сохранённой Б1; до прогона, HOTA ещё нет'))
    write(ROOT / 'experiments/project_history.json', catalog)
    from scripts.b123_safe_journal_20261027 import journal
    journal(IDENT, 'Б1: зарегистрирована рабочая интеграция в полный трекер',
        '*Постановка.* ' + p['action'] + '\n\n' + p['fixed'] + '\n\n*Критерии до прогона.* ' + p['technical'] + ' ' + p['primary'] +
        '\n\n*Стоимость.* ' + p['cost_gate'] + '\n\n*Сильные возражения и пределы.* ' + p['counterargument'] + ' ' + p['limitation'] +
        '\n\n*Сделано до расчёта.* Новый реальный адаптер, четыре CPU-проверки: выключенный путь/состояние, запрет будущего изображения, контекст исходного назначения, воспроизведение реальных признаков/вероятностей первой сохранённой строки. Старые экспорт/обучение не повторялись. Ошибка ожидания внешнего padding в тесте обнаружена до заморозки: значения padded Gtag различались между split12/18 и не передаются identity-модели; реальные входы и точные вероятности проверены. Источник: runs/recovery_research_20261027/b1_tracker/protocol.json. Полной HOTA пока нет.')


def publish():
    assert (RUN / 'b1_tracker_catalog_checks.json').exists()
    assert read(RUN / 'b1_tracker_catalog_checks.json')['passed']
    for name, folder, script in [('270_b1_full_tracker', 'b1_tracker', 'scripts/run_b1_tracker_20261028.py'),
                                 ('275_b1_full_tracker_audit', 'b1_tracker_audit', 'scripts/audit_b1_tracker_20261028.py')]:
        target = RUN / 'ready_jobs' / (name + '.json'); assert not target.exists()
        write(target, dict(name=name, folder=(RUN / folder).relative_to(ROOT).as_posix(), script=script,
                           script_sha256=sha(ROOT / script), protocol_sha256=sha(RUN / folder / 'protocol.json')))


if __name__ == '__main__':
    publish() if '--publish' in sys.argv else prepare()
