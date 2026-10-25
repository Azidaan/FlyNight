"""Record final descriptive checks and architecture milestone without new experiments."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_temporal_identity_trial_20261025 import note
from scripts.z13_temporal_pair_readiness_20261025 import read,write,sha,now
BASE=ROOT/'runs/z13_temporal_identity_20261025'


def main():
    assert read(BASE/'training/recorded.json')['complete']
    assert not (BASE/'records_finalized.json').exists()
    card_path=ROOT/'experiments/project_history.json';catalog=read(card_path)
    card=next(c for c in catalog['experiments'] if c['id']=='z13_temporal_identity_readout_v1')
    for rel,claim in [
        ('training/action_diagnostic.json','Описательные числа полезных, вредных и неизвестных выбранных действий; повторные наблюдения по трём начальным состояниям, не уникальные запросы.'),
        ('training/recovery2/numerical_parity.json','Проверка пакетного исполнения независимых моделей: градиенты, потери, шаг Adam и неизменные правила выбора.'),
        ('training/recovery2/structural_stack_recovery3.json','Исправление технического ограничения классов без изменения научного протокола.'),
    ]:
        card['evidence'].append(dict(kind='file',ref=(BASE/rel).relative_to(ROOT).as_posix(),claim=claim))
    write(card_path,catalog)
    local={}
    for path in sorted(BASE.rglob('*')):
        if path.is_file() and (path.suffix in ('.pt','.npz') or path.name=='rows.json'):
            local[path.relative_to(ROOT).as_posix()]=dict(sha256=sha(path),bytes=path.stat().st_size)
    write(BASE/'local_data_manifest.json',dict(created=now().isoformat(),local_only=True,files=local,
        reason='Large feature tensors, checkpoints and row arrays stay local; primary reports and exact hashes are published.'))
    check=read(ROOT/'dissertation/architecture/validation.json')
    assert check['diagram_sheets']==2 and check['all_text_fits'] and not check['box_overlaps'] and not check['edges_through_unrelated_nodes']
    write(BASE/'architecture_review.json',dict(at=now().isoformat(),validation_sha256=sha(ROOT/'dissertation/architecture/validation.json'),
        overview_viewed=True,new_readout_fragment_viewed=True,bottleneck_sheet_viewed=True,
        negative_branch_not_green=True,recurrence_and_two_sheet_roles_retained=True,
        actual_claim_review=dict(method_distinguished_from_research_object=True,no_duplicate_claim=True,only_completed_work=True),
        images=[str(p) for p in ['dissertation/architecture/FlyVis_architecture_preview.png','dissertation/architecture/FlyVis_temporal_identity_detail.png','dissertation/architecture/FlyVis_bottlenecks_preview.png']]))
    counts=read(BASE/'training/action_diagnostic.json')['by_arm']['D']['counts']
    assert counts['beneficial']==99 and counts['harmful']==128
    note('завершение записей и архитектурная веха межкадрового считывания',
        '*Проверено.* `runs/z13_temporal_identity_20261025/training/action_diagnostic.json`: D — 99 полезных и 128 вредных выбранных действий; '
        'сумма по трём начальным состояниям обучения и пяти разбиениям, не уникальные запросы. '
        'Неизвестные выбранные случаи учитываются отдельно; эти числа не заменяют среднюю разность и интервал по видео.\n\n'
        '*Архитектурная веха.* Добавлен проверенный внутренний узел межкадровых соответствий реальных областей на лист1, '
        'обновлены статусы данных и одной отрицательной реализации на лист2. Источники и генераторы сохранены; '
        'два листа, отсутствие перекрытий, обрезки текста и прохода стрелок через чужие блоки проверены генератором. '
        'Общий вид, новый читаемый фрагмент и лист приоритетов просмотрены: `runs/z13_temporal_identity_20261025/architecture_review.json`. '
        'P8 остаётся частичной; подтверждённого улучшения HOTA или точной причины неудачи нет.\n\n'
        '*Техническое исполнение.* Первая последовательная оценка цены корректно остановила дальнейшее обучение по исходному бюджету. '
        'Отдельная проверенная пакетная реализация завершила все 90 моделей до того же исходного срока 00:09:32 МСК 26.10.2026; '
        '`runs/z13_temporal_identity_20261025/training/recovery2/controller_result.json`: завершение 25.10.2026 в 23:57:49 МСК, '
        'exit_code=0, all_owned_ended=true. Ошибки и прежний контроллер сохранены, завершённая первая модель повторно не обучалась. '
        'Параметры моделей, потери, число шагов, разбиения и критерии не изменялись.\n\n'
        '*Сохранение.* Крупные массивы и веса остаются локально с размерами и хешами в '
        '`runs/z13_temporal_identity_20261025/local_data_manifest.json`; первичные отчёты, протоколы, код и проверка входят в публикацию. '
        'Отрицательный результат этой реализации не закрывает линию FlyVis.')
    write(BASE/'records_finalized.json',dict(at=now().isoformat(),complete=True,models=90,primary_pass=False))


if __name__=='__main__':main()
