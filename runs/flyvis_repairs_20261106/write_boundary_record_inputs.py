"""Prepare an evidence-bound technical addendum; shared ledgers are not written."""
from pathlib import Path
from datetime import datetime, timedelta, timezone
import json

ROOT = Path(__file__).resolve().parents[2]
RUN = 'runs/flyvis_repairs_20261106'
OUT = ROOT/RUN
SUITE = RUN + '/integrated_1794000756588983700'

def read(name):
    return json.loads((ROOT/name).read_text(encoding='utf-8'))

def main():
    result = read(SUITE+'/result.json')
    assert result['passed'] and result['returncode'] == 0 and result['test_files'] == 50
    assert result['concurrently_changed_sources'] == []
    assert not result['runtime']['cuda_initialized_before'] and not result['runtime']['cuda_initialized_after']
    assert '822 passed, 2 skipped' in (ROOT/SUITE/'tests.log').read_text()
    load = read(RUN+'/loading_constraint_followup/after_report.json')
    assert load['stock_count'] == 50 and load['stock_activity_exact_before_after']
    assert load['old_new_transfer_state_exact'] and load['old_new_transfer_activity_exact']
    assert load['protected_files_unchanged'] and load['source_destination_and_payload_unchanged']
    motion = read(RUN+'/final_boundary_review/manifest.json')
    assert motion['tests_after']['passed'] == 17 and motion['native_point_vector_exact']
    temporal = read(RUN+'/temporal_numpy_addendum_v3/manifest.json')
    assert temporal['verification']['forward_exact'] and temporal['verification']['reverse_exact']
    assert temporal['tests']['after'] == '150 passed'
    now = datetime.now(timezone(timedelta(hours=3)))
    obtained = (
        'Совместная проверка после дополнительных исправлений: 822 теста пройдены, два пропущены, 50 файлов тестов; '
        f'CUDA не инициализирована, исходники во время проверки неизменны ({SUITE}/tests.log; result.json). '
        'Ограничения загрузчика приняли все 50 штатных наборов весов; обычная загрузка и перенос на одинаковом исходном состоянии '
        'совпали с прежним кодом точно (runs/flyvis_repairs_20261106/loading_constraint_followup/after_report.json). '
        'В новом временном пути восстановление числовых параметров NumPy с NaN/Inf теперь отвергается до изменения модели; '
        '150 связанных тестов пройдены (runs/flyvis_repairs_20261106/late_temporal_boundary/tests_after.log). '
        'Для привязки полей движения к видео и передачи устройства пройдены 17 адресных и 190 связанных проверок; '
        'поля настоящего Farneback до и после совпали точно (runs/flyvis_repairs_20261106/final_boundary_review/manifest.json; after_report.json).'
    )
    interpretation = (
        'Подтверждены воспроизводимые пропуски проверки входных данных, а не повреждение текущих штатных весов. '
        'Объявленные ограничения ядра и конечность результата преобразования его параметров проверяются до копирования. '
        'Некорректные числовые значения NumPy больше не проходят восстановление оптимизатора нового временного пути. '
        'Новые сохранённые поля движения проверяют путь и контрольную сумму видео; явно выбранное устройство CPU достигает конструктора FlyVis.'
    )
    decision = (
        'Сохранить только узкие проверенные исправления, дополнительный обратимый патч V3 поверх опубликованного V2 и первичные результаты. '
        'Геометрию, формулу динамики, историческую временную схему и штатные обученные веса этим дополнением не менять. '
        'Запись в журнал и каталог добавляется отдельно без замены прежнего итога на 747 тестов. '
        'Публикацию выполнять в действующую основную ветку codex/system-integrity-20261106-e8f674ce; main оставить дополнительной копией по решению автора.'
    )
    limitation = (
        'Прирост HOTA и отсутствие всех ошибок не установлены. Проверка конечности преобразованных параметров не гарантирует конечность '
        'их последующих произведений и всей динамики. Исторические поля движения без сведений об исходном видео остаются непроверенными '
        'по происхождению; запись пары NPZ/JSON не стала общей транзакцией. Строковые и объектные массивы оптимизатора не входят в узкий '
        'контракт числового NumPy. Произвольное ограничение огромных временных разрывов не введено. Научное переобучение, перенос '
        'исторического З13 на другое время и полная оценка трекинга остаются невыполненными.'
    )
    evidence = [
        (SUITE+'/result.json', 'Совместная адресная проверка после дополнения'),
        (SUITE+'/tests.log', 'Число успешных и пропущенных тестов'),
        (RUN+'/loading_constraint_followup/after_report.json', 'Штатные веса, точное совпадение и сохранность при отказе'),
        (RUN+'/loading_constraint_followup/report.json', 'Воспроизведение ошибок ограничений и экспоненты'),
        (RUN+'/final_loading_constraint_review/report.json', 'Независимый пересчёт массивов и проверка контрольных сумм'),
        (RUN+'/temporal_numpy_addendum_v3/manifest.json', 'Обратимый дополнительный патч проверки оптимизатора'),
        (RUN+'/late_temporal_boundary/optimizer_after_report.json', 'Отказ до изменения весов и безопасный следующий шаг прежнего оптимизатора'),
        (RUN+'/final_boundary_review/manifest.json', 'Привязка полей к видео и передача явно выбранного устройства'),
        (RUN+'/late_temporal_boundary/time_report.json', 'Проверенные временные границы и незакрытые ограничения ресурсов'),
    ]
    card = dict(id='flyvis_system_boundary_addendum_20261107',
        title='Защита восстановления весов и оптимизатора; привязка полей движения к видео',
        area='infra', state='infrastructure', started='2026-11-07', done=now.strftime('%Y-%m-%d'),
        question='Проходят ли повреждённые числовые параметры и чужие поля движения через ранее проверенные границы?',
        hypothesis='Адресная техническая проверка граничных случаев исполнения; научная гипотеза о качестве трекинга не заявляется.',
        expected='Отказ до изменения состояния при подтверждённом дефекте, точное сохранение проверенного корректного исполнения.',
        acceptance=[dict(criterion='Воспроизведение исходного дефекта, отказ до изменения и контроль корректного исполнения. Критерий технического отчёта записан после проверок, предварительной научной регистрацией не является.',
                         met=True, evidence=RUN+'/loading_constraint_followup/after_report.json; '+RUN+'/temporal_numpy_addendum_v3/manifest.json; '+RUN+'/final_boundary_review/manifest.json'),
                    dict(criterion='Общая адресная проверка завершилась без ошибок и изменения исходников.', met=True, evidence=SUITE+'/result.json')],
        obtained=obtained, interpretation=interpretation, decision=decision, cannot_claim=limitation, metrics=[],
        evidence=[dict(kind='json' if path.endswith('.json') else 'file', ref=path, claim=claim) for path, claim in evidence])
    journal = (f'\n\n**{now:%d.%m.%Y, %H:%M} MSK** — Дополнительная проверка числовых параметров и происхождения движения\n\n'
               f'*Постановка.* {card["question"]} {card["expected"]}\n\n'
               f'*Получено (`{card["id"]}`).* {obtained}\n\n'
               f'*Вывод.* {interpretation} {limitation}\n\n'
               f'*Решение.* {decision}\n')
    report = ('Дополнение к FINAL_AUDIT_REPORT.txt, 07.11.2026\n\n1. Проверенный результат\n\n'+obtained+
              '\n\n2. Изменение поведения\n\n'+interpretation+'\n\n3. Решение\n\n'+decision+
              '\n\n4. Границы\n\n'+limitation+'\n\n5. Первичные результаты\n\n'+
              '\n\n'.join(path+' — '+claim for path, claim in evidence)+'\n')
    for name, raw in [('boundary_card.json', (json.dumps(card,ensure_ascii=False,indent=2)+'\n').encode()),
                      ('boundary_journal.txt', journal.encode()), ('BOUNDARY_ADDENDUM_REPORT.txt', report.encode())]:
        assert not (OUT/name).exists()
        (OUT/name).write_bytes(raw)
    review = dict(text_kind='Technical implementation record, no new scientific theory',
        checks=dict(method_vs_object='Changes refer to model loading, optimizer recovery and video cache; no algorithm is called a new research object.',
                    duplication='Only new boundary findings are added; the earlier final record is retained and referenced.',
                    actual_vs_planned='All numerical outcomes require primary evidence above; unperformed training and HOTA are explicitly excluded.'),
        all_sentences_reviewed=True)
    (OUT/'boundary_text_review.json').write_text(json.dumps(review,indent=2),encoding='utf-8')
    print(card['id'])

if __name__ == '__main__':
    main()
