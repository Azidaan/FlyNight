"""Record only a completed independently audited two-input experiment."""
import sys,json
from pathlib import Path
from datetime import datetime,timezone,timedelta
import psutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.flyvis_two_frame_input_20261107 import OUT,IDENT,read,sha,write

def main():
    a=read(OUT/'independent_audit.json'); e=read(OUT/'exposure_audit.json'); c=read(OUT/'controller.json')
    recovery=read(OUT/'recovery1_controller.json')
    assert c['passed'] is False and c['failed']=='audit' and recovery['passed']
    assert recovery['original_controller_sha256']==sha(OUT/'controller.json')
    assert a['passed'] and e['passed'] and a['pairs']==e['count']==172
    assert a['protocol_sha256']==sha(OUT/'protocol.json')
    owned=dict(c['owned']);launch=read(OUT/'launch.json');owned[str(launch['pid'])]=launch['birth']
    g=Path(read(OUT/'inference_completed.json')['guard']['supervisor_report']); guard=read(g)
    assert guard['reason'] is None and guard['returncode']==0
    owned.update({str(k):v for k,v in guard['owned'].items()})
    sup=read(g.parent/'supervisor.json');owned[str(sup['pid'])]=sup['birth']
    for pid,birth in owned.items():
        try:
            assert psutil.Process(int(pid)).create_time()!=birth,('owned_still_alive',pid)
        except psutil.NoSuchProcess:
            pass
    primary=[x for x in a['comparisons'] if x['primary']]
    state='completed_positive' if a['main'] else ('completed_negative' if all(x['ci97_5_pp'][1]<0 for x in primary) else 'inconclusive')
    values={x['name']:x['correct'] for x in a['conditions']}
    fmt=lambda v:f'{v:.3f}'.replace('.',',')
    descriptions=[f"{x['name']}: {fmt(x['delta_pp'])} п.п.;97,5% ДИ[{fmt(x['ci97_5_pp'][0])};{fmt(x['ci97_5_pp'][1])}], исправлено{x['gains']}, испорчено{x['losses']}" for x in primary]
    summary='Сигнал двух последних входов достиг T4 и T5. '+('Оба критерия выбора пройдены.' if a['main'] else 'Общий критерий улучшения выбора не пройден.')
    negative=[x['name'] for x in primary if x['ci97_5_pp'][1]<0]
    if negative:
        summary+=' Подтверждено ухудшение: '+', '.join(negative)+'.'
    text='# Два последних входа при общей ранней памяти\n\n'+summary+'\n\n'
    text+='Прежние172 пары,23 видео,26 исходных кластеров. Состояние после общей ранней предыстории клонировано без изменения; различаются только последние два уже известных входа. Они проходят прежние четыре внутренних обновления, дополнительных шагов и будущих кадров нет. Фиксированные окна шире рамок в1,25 раза, прежний масштаб и положение, нейтральный фон вне окна.\n\n'
    first=read(OUT/'first_case.json')
    text+=f"Первая фиксированная пара{first['uid']}: максимальное изменение нормированной карты T4={first['t4_max']:.9g}, T5={first['t5_max']:.9g}; тождественный контроль={first['identity_max']:.9g}. Порог прохождения1e-7 записан до расчёта. Источник:first_case.json. В полном наборе изменение выше этого порога: T4 — {e['t4_changed']}/172, T5 — {e['t5_changed']}/172; источник:exposure_audit.json. Это факт прохождения сигнала, не доказательство правильного распознавания машины.\n\n"
    text+='| Условие | Верных из172 | Доля верных |\n|---|---:|---:|\n'
    for x in a['conditions']:
        text+=f"| {x['name']} | {x['correct']:g} | {fmt(x['accuracy']*100)}% |\n"
    text+='\n'+'. '.join(descriptions)+'. Источник:independent_audit.json,conditions/comparisons. Обычные95% интервалы и описательное union-common сохранены там же. Тождественный контроль и воспроизведение старого общего выхода прошли прежние допуски; баллы пересчитаны отдельным оценщиком из сырых карт. Код оценщика общий с предыдущим опытом, вычисление баллов независимо от GPU-исполнителя.\n\n'
    text+='Меняется ограничение входа полного модуля FlyVis, ядро и исходный считыватель неизменны. Искусственная граница окна может вносить собственный сигнал движения. После общей предварительной обработки остаются перекрывающиеся части объектов;129/344 рамок частично за пределами старого поля. Это не проверка отдельных центрированных видов машин целиком, новой архитектуры считывания или полного трекера/HOTA. Исторические эпизоды уже использовались в исследовании.\n\n'
    text+='Исходная дополнительная сверка величины воздействия отказала при прямом сравнении вычитания float32 и float64. Контроллер с кодом1 сохранён. Отдельный recovery1 побитно подтвердил исходную32-битную статистику и независимо пересчитал64-битную; критерий прохождения сигнала1e-7 не изменён. GPU и полный пересчёт выбора не повторялись. Источники:exposure_recovery1_protocol.json,recovery1_controller.json.\n\n'
    text+=f"Относительно общего контроля доля верных меняется на{fmt(100*(values['separate']/values['common']-1))}%; измеренного потолка нет.\n"
    (OUT/'РЕЗУЛЬТАТ.md').open('x',encoding='utf8').write(text)
    now=datetime.now(timezone(timedelta(hours=3)));cat=read('experiments/project_history.json')
    card=next(x for x in cat['experiments'] if x['id']==IDENT);assert card['state']=='ready' and not card.get('obtained')
    card.update(state=state,done=now.strftime('%Y-%m-%d'),summary=summary,
        obtained=f"Общая карта{values['common']:g}/172, два окна вместе{values['union']:g}/172, раздельные{values['separate']:g}/172. "+'. '.join(descriptions)+f'. Источник:{OUT}/independent_audit.json,conditions/comparisons. '+f"T4 изменились{e['t4_changed']}/172, T5{e['t5_changed']}/172; источник:{OUT}/exposure_audit.json.",
        interpretation='В отличие от вмешательства только в текущий кадр, два последних входа меняют T4/T5. '+('Выбор улучшился по обоим критериям.' if a['main'] else 'Изменение активности не дало подтверждённого преимущества по обоим зарегистрированным критериям.'),
        decision='Сохранить все сравнения; не увеличивать число кадров и не менять считыватель по итогам этих эпизодов без отдельной согласованной постановки.')
    card['acceptance'][0].update(met=True,evidence=f'{OUT}/first_case.json;{OUT}/exposure_audit.json;{OUT}/independent_audit.json;{OUT}/controller_verified.json')
    card['acceptance'][1].update(met=a['main'],evidence=f'{OUT}/independent_audit.json,comparisons/main')
    card['evidence'].append(dict(kind='file',ref=(OUT/'РЕЗУЛЬТАТ.md').as_posix(),claim='Полный итог и ограничения'))
    write('experiments/project_history.json',cat)
    journal=Path('dissertation/ЕДИНЫЙ_ПЛАН.md');old=journal.read_text(encoding='utf8')
    entry=f'\n**{now:%d.%m.%Y, %H:%M} MSK** — `{IDENT}` завершён. *Постановка.* Последние два входа при общей ранней памяти. *Получено.* {card["obtained"]} *Вывод.* {card["interpretation"]} *Решение.* {card["decision"]} Отдельно прочитана статья Порцева–Макаренко оΣΠ-нейроне; это кандидат считывателя, не решение подачи окон. Оценка:`research_sigma_pi_20261107/ОЦЕНКА.md`; новый опыт по статье не запускался.\n'
    journal.write_text(old.replace('## Журнал изменений','## Журнал изменений\n'+entry,1),encoding='utf8')
    road=Path('FLYVIS_TRACKER_RESEARCH_ROADMAP.md')
    with road.open('a',encoding='utf8') as stream:
        stream.write(f'\n\n### {now:%d.%m.%Y %H:%M} MSK — Два входа при общей памяти\n\n{summary} {card["obtained"]} Это диагностика полного модуля, не HOTA. {card["decision"]}\n')
    write(OUT/'controller_verified.json',dict(passed=True,all_owned_ended=True,owned=owned,audit_sha256=sha(OUT/'independent_audit.json'),exposure_audit_sha256=sha(OUT/'exposure_audit.json'),state=state))
    print(summary,flush=True)

if __name__=='__main__':main()
