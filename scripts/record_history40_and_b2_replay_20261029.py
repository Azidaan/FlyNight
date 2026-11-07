"""Record verified matrix result and preserve all failed B2 numerical checks."""
from pathlib import Path
import sys,json,hashlib,shutil
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261028 import RUN,service

def main():
    out=RUN/'history40_agent_review';out.mkdir(exist_ok=False)
    names=['history_tracker_author_continuation1','history_tracker_author_audit1','history_tracker_author_record1']
    controllers={n:service.read(RUN/n/'controller_result.json') for n in names}
    for c in controllers.values():
        assert c['exit_code']==0 and c['all_owned_ended']
        assert not any(service.matching(k,v) for k,v in c['owned'].items())
    g=service.read(RUN/names[0]/'report.json');a=service.read(RUN/names[1]/'report.json')
    analysis=service.read(RUN/names[2]/'analysis.json');record=service.read(RUN/names[2]/'report.json')
    assert g['complete'] and a['passed'] and a['full_series'] and record['checks_passed']
    assert len(g['completed'])==len(a['completed'])==40
    assert {x['video'] for x in g['completed']}=={x['video'] for x in a['completed']}
    assert len(analysis['primary'])==12 and not analysis['main_acceptance']
    proof=dict(controllers=controllers,full_gpu_and_audit_video_sets_equal=True,unique_videos=40,
               events=a['events'],all12_criteria=analysis['primary'],main_acceptance=False,
               audit_sha256=service.sha(RUN/names[1]/'report.json'),
               gpu_report_sha256=service.sha(RUN/names[0]/'report.json'),
               record_sha256=service.sha(RUN/names[2]/'report.json'),updated=service.now())
    service.write(out/'verified.json',proof)
    path=ROOT/'experiments/system_problems.json';shutil.copy2(path,out/'system_problems_before.json')
    problems=service.read(path)
    evidence=[dict(source=(RUN/names[1]/'report.json').relative_to(ROOT).as_posix(),sha256=proof['audit_sha256'],claim='Полный независимый аудит40/32/3017196событий'),
              dict(source=(RUN/names[2]/'analysis.json').relative_to(ROOT).as_posix(),sha256=service.sha(RUN/names[2]/'analysis.json'),claim='Все12первичных сравнений, общий критерийнепройден')]
    p=next(x for x in problems['problems'] if x['id']=='С5')
    p['cases'].append(dict(date='2026-10-29',fact='Матрица1/3/4полностью проверена на40историческихval,32трекерах/3начальныхзначениях/2эпохах: тройнойвариант не превзошёлвсеконтроли,seed43хужеnativeи0послепоправки. Это не доказывает,что ошибки записи памяти — причина; более полное представление само по себе не решило задачу.',evidence=evidence))
    p['evidence'].extend(evidence)
    p['next_check']='Готовую матрицу не повторять. Следующая рабочая версия — отделить надёжную опору личности от неоднозначных новых назначений; минимально проверить правило записи памяти против согласованного простого контроля, с вредом при реальной смене внешности. Причинный эффект пока не измерен.'
    p['updated']='2026-10-29'
    c=next(x for x in problems['problems'] if x['id']=='С6')
    ev=[]
    for f in ['b2_sparse_training_audit_recovery3/controller_result.json','b2_loss_device_profile/report.json','b2_predictor_device_profile_recovery1/report.json','b2_predictor_cpu_profile/report.json','b2_training_gpu_predictor_audit/controller_result.json']:
        ev.append(dict(source=(RUN/f).relative_to(ROOT).as_posix(),sha256=service.sha(RUN/f),claim='Последовательность численных проверок и сохранённых отказов Б2'))
    c['cases'].append(dict(date='2026-10-29',fact='Б2: CPUзагрузкаисправна,но первыйdevlossрасходится0.000252>3e-5. GPUсручнойинициализациейYOLOтоже расходится; исходныйpredictorнаGPUпрошёл60потерьпервоговидео(max5.722e-6),но полныйповторостановилсянаDвтретьемвидео(5.531e-5). CPUсpredictorтоже недостигает3e-5. Не приписыватьвсёодномуустройству. Network.clampзависитотrequires_grad;идётотдельнаяпроверкасохраненияфлаговD. Всеотказысохранены,допускнеизменён,Б2valнедопущена.',evidence=ev))
    c['evidence'].extend(ev);c['updated']='2026-10-29'
    service.write(path,problems)
    stamp=datetime.now(timezone(timedelta(hours=3))).strftime('%d.%m.%Y, %H:%M MSK')
    text=f'''**{stamp}** — Полная матрица истории проверена; Б2 ограничена численным воспроизведением

Получено. history40_agent_review/verified.json:40уникальныхвидео/32трекера,3017196событийполногоCPUаудита,всеGPU/аудит/оформлениеконтроллерыcode0/all_owned_endedиPID+birthфактическизакончились. Все12первичныхсравненийпрочитаны. history_tracker_author_record1/analysis.json:общийкритерийfalse;парнаяHOTA134−native:seed42+0,029258с95%ДИ[−0,044785;0,102887],seed43−0,136858[−0,213970;−0,063440],seed44+0,013835[−0,065290;0,092390]. Уseed43скорректированнаяграницакnativeотрицательна[−0,245887;−0,031018];варианттакжехуже0. Карточкаinconclusive,история2эпохи/3сида/40историческихval,ядраFlyVisвветвинет. Польза полнойкомбинациинеподтверждена; причинанепрохождениянеустановлена. КаталогOK/58passed115,72с/граф53видимых1361файлсохранён,защищённыегипотезыпобайтны:history_tracker_author_record1/check_*.log.

Б2. Старый129CPUотказ/130зависимостьсохранены,обучениенеповторялось. Новые139/140/141/142диагностикии143GPUаудитсохраненыотдельно. ТочныйpredictorGPUпрошёл60потерьпервоговидеоспрежним3e-5;143полныйповторотказалнаDтретьеговидео5,531e-5. Это ещёнеполныйаудит/неHOTA. Обнаруженконкретныйстык:официальныйNetwork.clampприменяетграницыисимметриютолькокrequires_gradпараметрам,авоценкепослеобученияDэтифлагиоставалисьTrue;runtimeихснимает. Новая145проверяеттотжепрямойпроходвno_gradссохранениемфлаговD,безoptimizer/backward. Эторабочеепричинноеобъяснение,неподтверждённыйитог. 144полныйтрекернепрошёлзависимость,146/149/152НЕзапускались. Нельзяослаблять3e-5илипродвигатьБ2послепроваленногоаудита.

Решение. КартаПроблем:вС5добавленполныйитогматрицыиразличающаяпроверкаправилазаписипамяти,вС6—фактическаяцепочкачисленныхотказов. Общуюпричинувсехнеудачнеобъявлятьдоказанной. Продолжитьразрешённуюработупослепроверки145иостаткадоsave07:30/stop07:40;готовыеБ1/матрицу/обучениянеповторять.

'''
    plan=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';body=plan.read_text(encoding='utf-8');marker='## Журнал изменений';assert body.count(marker)==1
    plan.write_text(body.replace(marker,marker+'\n\n'+text,1),encoding='utf-8')
    with (RUN/'HANDOFF.md').open('a',encoding='utf-8') as f:f.write('\n\n'+text)
    print(out)

if __name__=='__main__':main()
