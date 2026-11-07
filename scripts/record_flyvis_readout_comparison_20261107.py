"""Record a completed cached-head comparison; a passed gate retains tracker work."""
import sys
from pathlib import Path
from datetime import datetime,timezone,timedelta
import psutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.flyvis_readout_comparison_20261107 import OUT,IDENT,read,write,sha

def main():
    a=read(OUT/'independent_audit.json');c=read(OUT/'controller.json');p=read(OUT/'protocol.json')
    assert c['passed'] and a['passed'] and a['protocol_sha256']==sha(OUT/'protocol.json')
    owned=dict(c['owned']);launch=read(OUT/'launch.json');owned[str(launch['pid'])]=launch['birth']
    for pid,birth in owned.items():
        try:assert psutil.Process(int(pid)).create_time()!=birth,('owned_still_alive',pid)
        except psutil.NoSuchProcess:pass
    names={'baseline':'Прежняя сумма','linear':'Линейный считыватель','mlp':'Обычная небольшая сеть','products':'Заданные произведения','sigma_pi':'ΣΠ-нейрон'}
    chosen=a['selected']['kind'];primary=next(r for r in a['primary'] if r['name']==chosen+'-baseline')
    state='completed_positive' if a['main'] else ('completed_negative' if primary['corrected_ci_pp'][1]<0 else 'inconclusive')
    fmt=lambda v:f'{v:.3f}'.replace('.',',')
    desc=lambda x:f"{names[x['name'].split('-')[0]]} минус {names[x['name'].split('-')[1]]}: {fmt(x['delta_pp'])} п.п.; скорректированный интервал [{fmt(x['corrected_ci_pp'][0])}; {fmt(x['corrected_ci_pp'][1])}], исправлено {x['gains']}, испорчено {x['losses']}"
    summary=f'Выбранный только на наборе подбора считыватель — {names[chosen]}. '+('Критерий перехода к полному трекеру пройден.' if a['main'] else 'Критерий перехода к полному трекеру не пройден.')
    negatives=[x['name'] for x in a['primary'] if x['corrected_ci_pp'][1]<0]
    if negatives:summary+=' Подтверждено ухудшение: '+', '.join(negatives)+'.'
    result='# Считыватели сохранённых T4/T5 при неизменном ядре\n\n'+summary+'\n\n'
    result+='Обучение — '+str(a['fit_counts']['fit84']['pairs'])+' пар из отдельной группы fit84; подбор — '+str(a['fit_counts']['fit20']['pairs'])+' пар fit20; оценка — '+str(a['pairs'])+' пар, '+str(a['contributing_videos'])+' видео с парами из 26 исходных кластеров. Это прежние исследовательские видео. Вход каждого кандидата — восемь вкладов активности T4/T5, умноженной на прошлые коэффициенты образца цели. Все варианты получают ровно эти числа; геометрия и разметка не входят в предсказатель.\n\n'
    result+='Ядро FlyVis, детектор, кэши и пространственное усреднение не менялись. Нового запуска ядра, видео или GPU не было. Нынешний балл равен сумме вкладов; обучаемые остаточные добавки первоначально равны нулю. У вариантов 9/21/37/19 параметров (linear/mlp/products/sigma_pi); 3 seed и 3 одинаковых уровня регуляризации, по 600 обновлений. Это сравнимые малые варианты, но число параметров не строго одинаково.\n\n'
    result+='| Считыватель | Верных выборов | Доля |\n|---|---:|---:|\n'
    for x in a['conditions']:result+=f"| {names[x['name']]} | {x['correct']:g} / {a['pairs']} | {fmt(x['accuracy']*100)}% |\n"
    result+='\nИсточник: independent_audit.json, conditions.\n\n'
    result+='## Основные сравнения\n\n'
    for x in a['primary']:result+=desc(x)+'.\n\n'
    result+='Источник: independent_audit.json, primary. Интервалы 98,75% учитывают четыре основных сравнения; обычные95% тоже сохранены. Парный повтор по исходным26 видео, 20000 выборок, seed42. Главное решение касается выбранного заранее варианта, а не лучшего по результатам оценки. Его три изменения по seed: '+str(a['seed_effects_pp'][chosen])+' п.п.\n\n'
    result+='## Что относится именно к ΣΠ\n\n'
    for x in a['sigma_secondary']:result+=desc(x)+'.\n\n'
    result+='Здесь отдельная семья трёх сравнений, интервалы98,3333%. Отдельная польза конструкции ΣΠ по всем трём сравнениям '+('подтверждена в этом опыте.' if a['sigma_specific_benefit'] else 'не подтверждена в этом опыте.')+' Это не доказательство равенства архитектур.\n\n'
    result+='## Проверки и ограничения\n\n'
    if (OUT/'descriptive_fit.json').exists():
        descriptive=read(OUT/'descriptive_fit.json')
        result+='Дополнительное описательное сравнение после основного результата, без подбора: доли верных на обучении/подборе/оценке.\n\n'
        for key,row in descriptive['conditions'].items():
            result+=names[key]+': '+', '.join(fmt(row[s]['accuracy']*100)+'%' for s in ['fit84','fit20','evaluate26'])+'.\n\n'
        result+='Источник: descriptive_fit.json. Различие наборов не доказывает единственную причину неудачи. Эти числа не меняют основной критерий и выбранную модель.\n\n'
    result+=f"Независимый оценщик NumPy повторил все предсказания и выборы; максимальное отличие балла {a['maximum_numpy_torch_difference']:.9g} при заранее записанном допуске1e-9. Старый поднабор172 пар воспроизвёл104 верных. Метки независимо сверены с прежними исходными строками. Выбор модели и регуляризации восстановлен по dev, файл selection.json был сохранён до первого чтения оценочных входов исполнителем обучения. Источник: independent_audit.json.\n\n"
    result+='Это диагностическая точность среди пар с известным верным кандидатом. Она не измеряет пропуски, неизвестные объекты, изменение последующих состояний и полные траектории. Новая голова не восстанавливает потерянную пространственным усреднением информацию. Нельзя по этому результату заявлять улучшение HOTA или отдельный вклад ядра относительно входа глаза.\n\n'
    baseline=next(r for r in a['conditions'] if r['name']=='baseline')['accuracy'];selected_value=next(r for r in a['conditions'] if r['name']==chosen)['accuracy']
    result+='Относительно прежнего считывателя доля верных у заранее выбранного варианта изменилась на '+fmt(100*(selected_value/baseline-1))+'%; измеренного потолка нет.\n\n'
    decision=('Подготовить и выполнить разрешённый отдельный согласованный опыт полного трекера с выбранным ансамблем, прежним считывателем и контролями; до этого не заявлять улучшение трекинга.' if a['main'] else 'Полный трекер для этой версии не запускать. Не заменять победителя по результатам оценочного набора и не подбирать ещё варианты на этих примерах. Сохранить все неудачные сравнения; считать эту малую проверку законченной.')
    result+='Решение: '+decision+'\n'
    with (OUT/'РЕЗУЛЬТАТ.md').open('x',encoding='utf8') as stream:stream.write(result)
    now=datetime.now(timezone(timedelta(hours=3)));cat=read('experiments/project_history.json');card=next(x for x in cat['experiments'] if x['id']==IDENT)
    assert card['state']=='ready' and not card.get('obtained')
    card.update(state=state,done=now.strftime('%Y-%m-%d'),summary=summary,
        obtained='; '.join(f"{x['name']} — {x['correct']:g}/{a['pairs']}" for x in a['conditions'])+'. '+'. '.join(desc(x) for x in a['primary'])+f'. Источник: {OUT}/independent_audit.json, conditions/primary.',
        interpretation='При неизменном ядре сравнили только обучаемые остаточные сочетания одних и тех же восьми вкладов. '+summary+' Отдельный критерий преимущества ΣΠ: '+str(a['sigma_specific_benefit'])+'. Положительная условная точность не равна HOTA.',decision=decision)
    card['acceptance'][0].update(met=True,evidence=f'{OUT}/independent_audit.json; {OUT}/controller_verified.json')
    card['acceptance'][1].update(met=a['main'],evidence=f'{OUT}/independent_audit.json, main/primary/selected/seed_effects_pp')
    card['acceptance'][2].update(met=a['sigma_specific_benefit'],evidence=f'{OUT}/independent_audit.json, sigma_secondary')
    card['evidence'].append(dict(kind='file',ref=(OUT/'РЕЗУЛЬТАТ.md').as_posix(),claim='Все сравнения, решение о полном трекере и ограничения'))
    write('experiments/project_history.json',cat)
    problem_file=Path('experiments/system_problems.json');problems=read(problem_file);problem=next(x for x in problems['problems'] if x['id']=='С4')
    evidence=dict(source=(OUT/'independent_audit.json').as_posix(),sha256=sha(OUT/'independent_audit.json'),claim='Сравнение четырёх считывателей на фиксированных признаках; не результат полного трекера')
    problem['cases'].append(dict(date=now.strftime('%Y-%m-%d'),fact=summary+' '+card['obtained']+' Это проверка узла считывания; причина ограничения информации и качество полных траекторий не установлены.',evidence=[evidence]))
    problem['evidence'].append(evidence);problem['cards'].append(IDENT);problem['updated']=now.strftime('%Y-%m-%d');write(problem_file,problems)
    journal=Path('dissertation/ЕДИНЫЙ_ПЛАН.md');text=journal.read_text(encoding='utf8')
    entry=f'\n**{now:%d.%m.%Y, %H:%M} MSK** — Завершён `{IDENT}`. *Постановка.* Четыре считывателя на неизменных сохранённых вкладовT4/T5, выбор только по dev. *Получено.* {card["obtained"]} *Вывод.* {card["interpretation"]} *Решение.* {decision} Статья Порцева–Макаренко использована для формул6–8 считывателя; её арифметические результаты не перенесены на трекинг.\n'
    journal.write_text(text.replace('## Журнал изменений','## Журнал изменений\n'+entry,1),encoding='utf8')
    with Path('FLYVIS_TRACKER_RESEARCH_ROADMAP.md').open('a',encoding='utf8') as stream:stream.write(f'\n\n### {now:%d.%m.%Y %H:%M} MSK — Считыватели сохранённых T4/T5\n\n{summary} {card["obtained"]} {decision}\n')
    write(OUT/'controller_verified.json',dict(passed=True,all_owned_ended=True,owned=owned,audit_sha256=sha(OUT/'independent_audit.json'),state=state,conditional_tracker_required=a['main']))
    print(summary,flush=True)

if __name__=='__main__':main()
