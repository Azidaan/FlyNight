"""Summarize verified saved results, without any numerical experiment rerun."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261029 import RUN,service

def main():
    out=RUN/'cost_use_agent_review';record=RUN/'cost_use_agent_record'
    c=service.read(record/'controller_result.json');assert c['exit_code']==0 and c['all_owned_ended']
    assert not any(service.matching(k,v) for k,v in c['owned'].items())
    r=service.read(record/'report.json');assert r['complete'] and r['checks_passed'] and r['confirmed_positive_comparisons']==5
    graph=service.read(record/'graph_audit.json');assert graph['passed'] and graph['all_hypotheses_and_legacy_byte_identical']
    old=service.read(out/'verified.json');assert old['passed']
    assert all(service.sha(ROOT/x['source'])==x['sha256'] for x in old['controllers'].values())
    assert all(not any(service.matching(k,v) for k,v in service.read(ROOT/x['source'])['owned'].items()) for x in old['controllers'].values())
    src='runs/continuation_20261029/cost_use_val_audit/report.json';a=service.read(ROOT/src)
    assert all(a[k]==old[k] for k in ['primary_comparisons','native_comparisons','extractor_descriptive','pooled','main_acceptance'])
    card=next(x for x in service.read(ROOT/'experiments/project_history.json')['experiments'] if x['id']=='query_cost_use_v1')
    assert card['state']=='inconclusive' and len(card['confirmed_results'])==5
    proof=dict(complete=True,passed=True,source=src,source_sha256=service.sha(ROOT/src),
        verified_sha256=service.sha(out/'verified.json'),final_record_controller_sha256=service.sha(record/'controller_result.json'),
        final_record_actual_all_owned_ended=True,graph_sha256=service.sha(record/'graph_audit.json'),
        original_failed_preparation_preserved_sha256=service.sha(RUN/'cost_use_val_preparation/controller_result.json'),
        main_acceptance=False,confirmed_positive_comparisons=5,updated=service.now())
    service.write(out/'clarification_verified.json',proof)
    labels={'native':'Исходный трекер','r50_last':'R50, прежняя добавка','detic_last':'Detic, прежняя добавка',
        'r50_centered':'R50, центрированная поправка','detic_centered':'Detic, центрированная поправка','r50_shift':'Общий сдвиг без содержания внешности'}
    def number(x):return f'{x:.6f}'.replace('.',',')
    def interval(x):return '['+'; '.join(number(v) for v in x)+']'
    text='# Применение признаков: прибавка стоимости отделена от содержания внешности\n\n'
    text+='Полная проверка завершена: двадцать исторических видео, три новых полных трекера и отдельное повторение всех выходов, стоимости, памяти и метрик. Память последнего наблюдения, детекции, признаки, моменты запросов и исходные геометрические допуски сохранены. Старые научные результаты и точные GPU FP32-векторы переиспользованы по контрольным суммам; нового извлечения признаков и обучения не было.\n\n'
    text+='Центрирование улучшило прежнюю положительную добавку с обоими извлекателями. Однако центрирование не показало подтверждённого превосходства над общим сдвигом без содержания внешности. Общий критерий четырёх сравнений не пройден. Это не доказательство равенства методов.\n\n'
    text+='## 1. Итоговые метрики объединённых двадцати видео\n\n| Вариант | HOTA | AssA | IDF1 | FP | FN | Смены ID | HOTA к исходному, % |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
    for n,m in a['pooled'].items():
        text+='| '+labels[n]+' | '+' | '.join([number(m['HOTA']),number(m['AssA']),number(m['IDF1']),str(m['CLR_FP']),str(m['CLR_FN']),str(int(m['IDSw'])),number(old['relative_hota_percent'][n])])+' |\n'
    text+='\nHOTA — совместная оценка обнаружения и связывания; AssA — качество связывания; IDF1 — согласованность идентификаторов; FP/FN — ложные обнаружения и пропуски. Неблагоприятный результат тоже сохранён: у новых вариантов FP больше исходного. Объединённая HOTA отличается по определению от средней парной разницы по видео.\n\n'
    text+='## 2. Заранее заданные первичные сравнения\n\n| Разница | Средняя парная HOTA | Обычный интервал 95% | Бонферрони 98,75% |\n|---|---:|---|---|\n'
    for key,v in a['primary_comparisons'].items():
        encoder='R50' if key.startswith('r50') else 'Detic';control='прежняя добавка' if key.endswith('_old') else 'общий сдвиг'
        text+=f'| {encoder}: центрирование − {control} | {number(v["mean_per_video"])} | {interval(v["ci95"])} | {interval(v["ci_bonferroni_9875"])} |\n'
    text+='\nПарное повторное формирование выборок: 20 000, фиксированное начальное значение1. Главный критерий требует положительных нижних границ всех четырёх скорректированных интервалов. Два сравнения с общим сдвигом включают ноль.\n\n'
    text+='## 3. Частные положительные результаты и границы вывода\n\n'
    text+='Сохранены пять частных положительных сравнений: два с прежней добавкой и три вторичных с исходным трекером. Все пять прошли дополнительно более строгую поправку на семью из семи сравнений, записанную агентом до пробного прогона; главный критерий четырёх сравнений не менялся. Вторичные сравнения с исходным трекером не подтверждают отдельную пользу содержания признаков.\n\n'
    for key,v in a['native_comparisons'].items():text+=labels[key]+': парная HOTA к исходному '+number(v['mean_per_video'])+', дополнительный интервал семьи7 '+interval(v['ci_global7'])+'.\n\n'
    text+='Подтверждено влияние способа применения стоимости в этой авторской системе. Сдвиг стоимости части допустимых пар сам даёт прибавку на исторических данных; опыт не доказывает единственный механизм только порога решателя. Наличие внешности сверх этого сдвига не обеспечило прохождение главного критерия. Это не полный Deep SORT/BoT-SORT и не вклад ядра FlyVis.\n\n'
    text+='Те же двадцать видео уже использовались в последовательном исследовании, поэтому это не новый независимый итоговый тест. Вес0,1 и середина0,5 не подбирались по качеству. Прежние девять выборочных сверок признаков переиспользованы по контрольным суммам; полного нового повторения всех пикселей не было. Доля измеренного потолка неизвестна: такой потолок не измерялся.\n\n'
    text+='## 4. Сохранённые доказательства\n\nПервичный независимый отчёт: `'+src+'`. Полная проверка входных/выходных контрольных сумм, повторного использования контролей и фактического конца всех процессов: `cost_use_agent_review/verified.json` и `clarification_verified.json`. Итоговое оформление: `cost_use_agent_record/report.json`, `graph_audit.json`, `check_0.log`, `check_1.log`. Все три новых научных трекера завершены; граф гипотез и старые P1 сохранены побайтно, удалений и перемещений нет. Служебная ошибка330 сохранена отдельно; исправление331 не изменило научные условия и не повторяло готовые результаты.\n'
    (out/'РЕЗУЛЬТАТ.md').write_text(text,encoding='utf-8')
    print(proof)

if __name__=='__main__':main()
