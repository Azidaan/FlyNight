"""Record completed fixed-policy HOTA from saved primary artifacts, without rerunning it."""
from datetime import datetime
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.z13_hota_deadline_guard import WindowsProcesses
from tracking.z13_checkpoint import atomic_json
RUN = ROOT / 'runs/z13_shared_masa_hota_v1_20261015/full'
SOURCE = 'runs/z13_shared_masa_hota_v1_20261015/full/eval/report.json'
VERIFY = 'runs/z13_shared_masa_hota_v1_20261015/full/final_verification.json'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            result.update(block)
    return result.hexdigest()


def ru(x, places=3):
    return f'{x:+.{places}f}'.replace('.', ',')


def main():
    now = datetime.now().astimezone()
    assert not (RUN / 'final_result_recorded.json').exists(), 'Result already recorded'
    report = read(ROOT / SOURCE); status = read(RUN / 'status.json')
    protocol = read(RUN / 'protocol.json'); identity = read(RUN / 'inference_identity.json')
    assert status['state'] == 'completed' and status['completed_jobs'] == status['total_jobs'] == 107
    assert report['videos'] == len(protocol['videos']) == 35
    assert report['pooled']['A_masa|all']['HOTA'] == 61.65477893761337
    assert report['criteria'] == {'implementation_D_vs_A': False, 'fly_added_D_vs_B': False, 'core_training_D_vs_C': False}
    assert report['diagnostic_policy'] and not report['independent_confirmation'] and report['old_90_percent_gate_unchanged']
    assert all(sha(ROOT / name) == value for name, value in identity.items()), 'Inference identity changed'
    markers = [read(x) for x in (RUN / 'completed_jobs').glob('*.json')]
    assert len(markers) == 107 and all(x['identity'] == identity for x in markers)
    expected_ids = {f'{arm}_sig_{video}' for arm in protocol['arms'] for video in protocol['videos']} | {'shared_masa', 'eval'}
    assert {x['job']['id'] for x in markers} == expected_ids
    for arm in protocol['arms']:
        for mode in ('masa', 'none'):
            assert {x.stem for x in (RUN / 'eval' / f'{arm}_{mode}').glob('*.json')} == set(protocol['videos'])
    guard = read(next(RUN.glob('deadline_guard_*/result.json')))
    assert guard['state'] == 'controller_completed'
    processes = WindowsProcesses()
    assert not any(processes.creation(int(pid)) == created for pid, created in guard['known_owned_processes'].items())
    details = [read(x) for x in (RUN / 'shared_masa_details').glob('*.json')]
    assert len(details) == 35 and all(x['parity_guard_checked'] for x in details)
    verification = {'verified_at': now.isoformat(), 'report_sha256': sha(ROOT / SOURCE),
        'completed_at': status['updated'], 'full_dev35_all_three_arms': True,
        'completed_jobs': 107, 'unchanged_input_code_hashes': len(identity),
        'all_output_video_sets_equal_protocol': True, 'all_own_processes_exited': True,
        'guard_result': 'controller_completed', 'MASA_first_frame_guards_passed': 35,
        'MASA_calls': sum(x['feature_calls'] for x in details),
        'MASA_calls_without_sharing': sum(x['feature_calls_without_sharing'] for x in details),
        'MASA_guard_calls': sum(x['guard_feature_calls'] for x in details),
        'MASA_sum_video_seconds': sum(x['seconds'] for x in details),
        'MASA_scope': 'Actual shared elapsed only; no full-set separate execution speed or bitwise parity claim.',
        'scientific_criteria': report['criteria'], 'no_new_GPU_or_TrackEval': True}
    plan = ROOT / 'dissertation/ЕДИНЫЙ_ПЛАН.md'; plan_text = plan.read_text(encoding='utf-8')
    rows = [line for line in plan_text.splitlines() if line.startswith('| З13 |')]
    assert len(rows) == 1
    verification['old_Z13_plan_row'] = rows[0]
    atomic_json(ROOT / VERIFY, verification)
    catalog_path = ROOT / 'experiments/project_history.json'; catalog = read(catalog_path)
    scientific = next(c for c in catalog['experiments'] if c['id'] == 'fly_spatial_hota_policy_z13_v1')
    execution = next(c for c in catalog['experiments'] if c['id'] == 'z13_shared_masa_execution_v1')
    assert scientific['state'] == execution['state'] == 'running'
    def set_final(card, field, value):
        if field in card and card[field] != value:
            card.setdefault('revisions', []).append({'date': now.strftime('%Y-%m-%d'), 'commit': '',
                'field': field, 'old': card[field], 'reason': 'Полный фиксированный опыт завершён; первичный результат заменяет предварительную формулировку без удаления истории.'})
        card[field] = value
    diff = report['diff']; base = report['pooled']['A_masa|all']['HOTA']
    d = diff['D_masa-A_masa|all']; b = diff['D_masa-B_masa|all']; c = diff['D_masa-C_masa|all']
    scientific.update(state='completed_negative', done=now.strftime('%Y-%m-%d'))
    set_final(scientific, 'obtained', f'Полный dev35, прежняя MASA, {SOURCE}: общая HOTA A={base:.12f}, B={report["pooled"]["B_masa|all"]["HOTA"]:.12f}, C={report["pooled"]["C_masa|all"]["HOTA"]:.12f}, D={report["pooled"]["D_masa|all"]["HOTA"]:.12f} (pooled.*_masa|all.HOTA). D−A: общий {ru(d["pooled_gain"],6)}, среднее по видео {ru(d["mean_per_video"],6)} [{ru(d["ci95"][0],6)}; {ru(d["ci95"][1],6)}]. D−B: общий {ru(b["pooled_gain"],6)}, по видео {ru(b["mean_per_video"],6)} [{ru(b["ci95"][0],6)}; {ru(b["ci95"][1],6)}]. D−C: общий {ru(c["pooled_gain"],6)}, по видео {ru(c["mean_per_video"],6)} [{ru(c["ci95"][0],6)}; {ru(c["ci95"][1],6)}]. Все три criteria=false. D вмешался 2131 раз из 209659 шагов; events.D. Ночь 15 видео: D−A общий −0,009502, по видео −0,023067 [−0,077479; +0,021147], diff.D_masa-A_masa|night. Источник всех чисел — указанный report.json; полнота/завершение — {VERIFY}.')
    set_final(scientific, 'interpretation', 'Зафиксированная политика верхнего 1% не дала подтверждённой пользы стенду, добавочного вклада FlyVis или пользы обучения ядра. Малый положительный общий прирост D−A ниже принятого минимума +0,10; все основные парные интервалы содержат ноль. Общий D−B положителен, средняя парная разность отрицательна: это разные способы усреднения. Причинный узел провала этим сравнением не установлен; нельзя приписывать его исключительно считыванию, представлению цели, входу или обучению.')
    set_final(scientific, 'cannot_claim', 'Нет подтверждения улучшения HOTA, ночной пользы, преимущества перед контролем без FlyVis или перед замороженным ядром. Также не доказаны отсутствие любого эффекта или вред. Dev многократно использован; независимые видео не открывались. Одна политика и одна реализация не опровергают пространственное считывание или модернизацию FlyVis вообще. Инфраструктурное ускорение MASA не является научным выигрышем мухи.')
    set_final(scientific, 'decision', 'Эта реализация с данной политикой не выполнила заранее записанные критерии. Завершить опыт без подбора новых порогов, без обучения и без открытия 75 отложенных видео. Не менять стенд. Более широкую линию FlyVis не закрывать; выбор следующего узла или новой реализации требует отдельного решения автора. Статусы проблем мухи не повышаются, причинная локализация провала не доказана. Наблюдение отключить после фиксации результата.')
    for i, key in enumerate(('implementation_D_vs_A', 'fly_added_D_vs_B', 'core_training_D_vs_C'), start=1):
        assert scientific['acceptance'][i]['met'] is None
        scientific['acceptance'][i].update(met=report['criteria'][key], evidence=SOURCE + ': criteria.' + key)
    scientific['metrics'] = [{'key': 'HOTA', 'value': report['pooled'][f'{arm}_masa|all']['HOTA'],
        'baseline': base, 'subset': 'dev35, после MASA, ' + arm, 'source': SOURCE,
        'locator': f'pooled.{arm}_masa|all.HOTA'} for arm in ('B', 'C', 'D')]
    for card in (scientific, execution):
        card.setdefault('evidence', []).extend([{'kind': 'json', 'ref': SOURCE, 'claim': 'Полный первичный HOTA, парные интервалы, события и критерии'},
                                               {'kind': 'json', 'ref': VERIFY, 'claim': 'Полнота 107 заданий, неизменность исходных контрактов и завершение собственного дерева'}])
    execution.update(state='infrastructure', done=now.strftime('%Y-%m-%d'))
    execution['acceptance'][2].update(met=True, evidence=VERIFY)
    set_final(execution, 'obtained', f'107/107 заданий завершены 15.10 в 05:23:50 MSK, status.json и {VERIFY}. Все три варианта по 35 видео и полная оценка сохранены; 135 хэшей исходного контракта не менялись, собственное дерево завершилось. На всех 35 видео прошла проверка равенства первого общего кадра. Признаки кадра: {verification["MASA_calls"]} вместо {verification["MASA_calls_without_sharing"]}, плюс {verification["MASA_guard_calls"]} контрольных вызовов; сумма времени совместных видео {verification["MASA_sum_video_seconds"]:.6f} с. Источник этих чисел — {VERIFY} и shared_masa_details/*.json. Предварительная скорость и полное побитовое равенство — runs/z13_shared_masa_probe_v1_20261015/report.json, на одном видео.')
    set_final(execution, 'interpretation', 'Разрешённый фиксированный опыт успешно доведён до полного HOTA с сохранением ранее готовых сигнатур. Техническое объединение MASA сократило число одинаковых вычислений кадра. На всём dev35 обычный отдельный MASA повторно не запускался; полная сравнительная скорость и побитовое равенство всему набору не измерены.')
    set_final(execution, 'cannot_claim', 'Успех исполнения и скорость технической пробы не означают улучшение научных метрик. Нет отдельного полного прогона старого исполнения для оценки ускорения всего HOTA или полного побитового сравнения; такие проверки не запускались.')
    set_final(execution, 'decision', 'Завершить инфраструктурное исполнение; GPU и независимый сторож уже штатно закончились до 07:40. Сохранить все результаты, отключить наблюдение; дальнейшие расчёты не запускать.')
    catalog_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    row_parts = rows[0].split('|')
    row_parts[-2] = ' проверены З13 и пространственная версия; 15.10 завершён `fly_spatial_hota_policy_z13_v1`: все три критерия HOTA на dev35 не выполнены, D−стенд общий +0,017911, парный +0,008680 [−0,029900; +0,056076]; `runs/z13_shared_masa_hota_v1_20261015/full/eval/report.json`. Это итог одной реализации и политики; линия не закрывается, новых запусков не назначено '
    plan_text = plan_text.replace(rows[0], '|'.join(row_parts), 1)
    entry = f'''**{now:%d.%m.%Y, %H:%M} MSK** — полный HOTA З13 завершён до срока, все три критерия не выполнены

*Постановка (`fly_spatial_hota_policy_z13_v1`).* Одна заранее заданная политика верхнего 1% train26: B без FlyVis, C с замороженной мухой, D с обученной мухой. Dev35, 30 Гц, прежняя сшивка MASA. Нового обучения, выбора порогов по dev и открытия отложенных видео не было; прежний провал допуска 90% сохранён.

*Получено.* `runs/z13_shared_masa_hota_v1_20261015/full/status.json`: 107/107, конец 05:23:50 MSK. `{SOURCE}`, pooled: общая HOTA A 61,654779; B 61,657628; C 61,655094; D 61,672690. D−A общий +0,017911; среднее по видео +0,008680 [−0,029900; +0,056076], лучше 10/35. D−B общий +0,015062, по видео −0,028130 [−0,151824; +0,070630], лучше 17/35. D−C общий +0,017596, по видео +0,007784 [−0,133805; +0,118780], лучше 15/35. Источник разностей — diff.*|all. Все criteria=false. Ночь: D−A общий −0,009502; парный −0,023067 [−0,077479; +0,021147], 15 видео, diff.D_masa-A_masa|night. D: 2131 вмешательство, 209659 шагов, events.D. Общий и парный приросты не смешиваются.

*Вывод.* Критерии пользы стенду, добавки FlyVis и обучения ядра не выполнены. D−A ниже практического минимума +0,10, все основные интервалы содержат ноль. Это отрицательный итог конкретной реализации и политики, не доказательство бесполезности FlyVis или вреда. Узел, причинно ответственный за провал, не установлен; статусы проблем мухи не повышаются. Подбор нового порога и новых реализаций после результата не проводился.

*Инфраструктура (`z13_shared_masa_execution_v1`).* `{VERIFY}`: 35 видео для каждой ветки, все 107 маркеров, 135 неизменных хэшей. Проверки первого общего кадра прошли на всех 35 видео. Вызовов общих признаков 29089 вместо 87060, плюс 35 контрольных; сумма сохранённых длительностей видео MASA 2168,796091 с, shared_masa_details. Отдельный полный старый MASA не пересчитывался: процент ускорения всего опыта не измерен. Сторож завершился 05:23:51; живых процессов собственного дерева нет, GPU освобождён, результаты сохранены. Прежняя проба скорости остаётся технической, не научным результатом.

*Решение.* Завершить этот опыт без изменения стенда, без новых GPU-расчётов и открытия отложенных видео; выбор следующей модернизации — отдельно с автором. Наблюдение отключается после записи. По общей HOTA D к A: {100*d['pooled_gain']/base:.6f}% (не подтверждено парным ДИ); доля оракула не указывается: согласованный оракул строго этой политики/действия после MASA не измерен. Git остаётся заблокирован существующей `.git/index.lock`, её не удалять; записи и результаты сохранены локально, коммит/push не состоялись.

'''
    # Preserve decimal commas in prose; do not alter primary paths.
    entry = entry.replace(f"{100*d['pooled_gain']/base:.6f}%", f"{100*d['pooled_gain']/base:.6f}".replace('.', ',') + '%')
    assert '## Журнал изменений\n\n' in plan_text
    plan.write_text(plan_text.replace('## Журнал изменений\n\n', '## Журнал изменений\n\n' + entry, 1), encoding='utf-8')
    atomic_json(RUN / 'final_result_recorded.json', {'at': now.isoformat(), 'scientific_state': scientific['state'],
                'execution_state': execution['state'], 'source': SOURCE, 'verification': VERIFY})
    print(json.dumps({'recorded': True, 'scientific_result': scientific['state'], 'all_criteria': report['criteria'],
                      'own_processes_exited': True}, ensure_ascii=False))


if __name__ == '__main__':
    main()
