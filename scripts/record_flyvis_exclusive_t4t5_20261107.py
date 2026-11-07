"""Record the prespecified outcome. Keeps failure of the primary criterion visible."""
from pathlib import Path
import sys
from datetime import datetime
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import flyvis_exclusive_t4t5_20261107 as run

def main():
    folder=run.RUN;r=run.load(folder/'report.json');p=run.load(folder/'protocol.json');m=run.load(folder/'measurement.json')
    ctrl=run.load(folder/'controller.json')
    assert ctrl['returncode']==0 and not ctrl['reason'] and r['measurement_sha256']==run.sha(folder/'measurement.json')
    import psutil
    assert not any(psutil.pid_exists(int(pid)) and psutil.Process(int(pid)).create_time()==birth for pid,birth in ctrl['owned'].items())
    key=r['status']; status={'improved':'completed_positive','worse':'completed_negative','inconclusive':'inconclusive'}[key]
    ss=r['comparison']['evaluate26'];q=ss['primary']; details=r['cohorts']['evaluate26']; reasons=r['labels']['evaluate26']
    num=lambda n:f'{n:.2f}'.replace('.',',')
    txt=(f"Основные 26 видео, ровно две допустимые рамки. Из {details['exactly_two']} запросов у {reasons['ok']} известна единственная верная рамка по строгим меткам. "
         f"У {q['n']} из них обе имеют собственные ячейки. Вся рамка: {q['whole_correct']} верных, {q['whole_ties']} ничьих, {num(q['whole_pct'])}%; "
         f"только свои ячейки: {q['unique_correct']} верных, {q['unique_ties']} ничьих, {num(q['exclusive_pct'])}%. "
         f"Разность {num(q['difference_pp'])} п.п., 95%-й парный интервал по видео [{num(q['ci95_pp'][0])}; {num(q['ci95_pp'][1])}].")
    interp={'improved':'При данном считывателе собственные ячейки улучшили выбор в проверенных парах.',
            'worse':'При данном считывателе исключение общих ячеек ухудшило выбор в проверенных парах.',
            'inconclusive':'По заранее записанному сравнению преимущество собственных ячеек не подтверждено; равенство методов также не установлено.'}[key]
    limits='Это выбор между двумя сохранёнными детекциями при заданной цели, а не изменение решений работающего трекера. Использованы нормированные каналы T4/T5 штатной модели и восемь ранее сохранённых коэффициентов наблюдаемой цели. Исторические видео, без обучения, нового прохода модели или HOTA. Уточнение формулировки protocol.inputs: прежний порог уверенности 0,25 отбирал детекции; личности присвоены отдельно, взаимно однозначным сопоставлением с разметкой внутри класса при IoU ≥ 0,5, а не выбором наибольшей уверенности. Использованы готовые метки без изменений.'
    decision='Сохранить результат диагностики. Изменение штатного сопоставления, новый подбор весов и обучение не выполнялись.'
    tables=['| Выборка | Пар | Вся рамка, % | Свои, % | Разность, п.п. | 95% по видео, п.п. |','|---|---:|---:|---:|---:|---|']
    for sp,g,name in [('evaluate26','primary','Основное сравнение'),('evaluate26','overlap_both_unique','Только частичное пересечение'),('evaluate26','disjoint','Без общих ячеек'),('evaluate26','exactly_two_fallback','Все размеченные пары, при пустоте прежнее правило'),('evaluate26','all_admitted_pairs_fallback','Все пары в том числе при трёх и более кандидатах, при пустоте прежнее правило'),('fit20','primary','Прежние 20 видео, описательно'),('fit84','primary','Прежние 84 видео, описательно')]:
        a=r['comparison'][sp][g]; tables.append(f"| {name} | {a['n']} | {num(a['whole_pct'])} | {num(a['exclusive_pct'])} | {num(a['difference_pp'])} | [{num(a['ci95_pp'][0])}; {num(a['ci95_pp'][1])}] |")
    label_text=f"Не оценены: неизвестная цель или прежний тип запроса — {reasons.get('unknown_target_or_query',0)}, неизвестная личность кандидата — {reasons.get('unknown_candidate',0)}, обе рамки соответствуют цели — {reasons.get('both_target',0)}, ни одна — {reasons.get('neither_target',0)}. У {details['labeled_no_exclusive']} размеченных пар хотя бы один собственный набор пуст; для них отдельно показан возврат к прежнему правилу."
    how=f"В основном сравнении ошибочный выбор стал верным {q['wrong_to_right']} раз, верный стал ошибочным {q['right_to_wrong']} раз. Изменения с ничьей входят в показатель с половинным весом. {q['videos_with_evaluable_pairs']} видео дали оцениваемые пары."
    source=(folder/'report.json').as_posix()
    prose='# Помогают ли собственные ячейки T4/T5 выбрать продолжение трека?\n\n'+txt+'\n\n'+how+'\n\n'+'\n'.join(tables)+'\n\n'+label_text+'\n\n'+interp+' '+limits+'\n\n'+decision+'\n\nИсточник: '+source+'. Проверка выбора двумя вычислительными формулами — measurement.json; по каждой паре — choices.csv.gz, по видео — per_video.json.\n'
    with (folder/'РЕЗУЛЬТАТ.md').open('x',encoding='utf-8') as f:f.write(prose)
    c=run.load(run.common.CAT);card=next(x for x in c['experiments'] if x['id']==run.ID);assert card['state']=='ready'
    card.update(state=status,done=datetime.now().strftime('%Y-%m-%d'),summary=interp,obtained=txt+' '+how+' Источник: '+source+' → comparison.evaluate26.primary.',interpretation=interp,decision=decision,cannot_claim=limits)
    card['acceptance'][0].update(met=True,evidence=(folder/'measurement.json').as_posix())
    card['acceptance'][1].update(met=key=='improved',evidence=source+' → comparison.evaluate26.primary')
    for n,title in [('report.json','Первичный итог и все сравнения'),('measurement.json','Сверка масок и независимый пересчёт баллов выбора'),('РЕЗУЛЬТАТ.md','Числа, смысл и ограничения')]:
        card['evidence'].append(dict(kind='file',ref=(folder/n).as_posix(),claim=title))
    run.dump(run.common.CAT,c);run.common.journal('выбор по T4/T5 в собственных ячейках (`'+run.ID+'`). *Получено.* '+txt+' '+how+' *Вывод.* '+interp+' *Решение.* '+decision+' '+source+'.')
    with (ROOT/'FLYVIS_TRACKER_RESEARCH_ROADMAP.md').open('a',encoding='utf-8') as f:
        f.write('\n## 07.11.2026 — T4/T5: исключение общей части конкурирующих рамок\n\n'+txt+' '+interp+' '+limits+' Источник: '+source+'.\n')
    print(txt)
if __name__=='__main__':main()
