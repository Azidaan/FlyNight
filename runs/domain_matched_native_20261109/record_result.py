"""Record the fixed native comparison only after metrics, fate and audit pass.

This writer never changes scientific code, heads, thresholds or section7 plans.
"""
from evaluation_common import *
start(900)
p=read(OUT/'protocol.json');result=read(OUT/'comparison_report.json');fate=read(OUT/'id_fate.json');audit=read(OUT/'runtime_audit.json')
assert result['passed'] and fate['passed'] and audit['passed'] and read(OUT/'prerun_checks.json')['passed']
assert result['evaluation_sha256']==fate['evaluation_sha256']==sha(OUT/'evaluation_protocol.json')
verify(read(OUT/'evaluation_protocol.json'));verify(read(OUT/'audit_protocol.json'));verify(read(OUT/'execution_protocol.json'))
assert set(fate['summary'])==set(p['arms']) and len(audit['receipts'])==100
accepted=bool(result['accepted_flyvis_system_and_increment']);now=datetime.now(MSK)
def atomic(path,data,before):
    temp=path.with_suffix(path.suffix+'.domain_result.pending');assert not temp.exists();temp.write_bytes(data)
    for _ in range(30):
        assert path.read_bytes()==before
        try:temp.replace(path);return
        except PermissionError:time.sleep(.1)
    raise PermissionError(str(path))
state='completed_positive' if accepted else 'completed_negative'
outcome='Полный разработческий пилот выполнил зарегистрированные критерии.' if accepted else 'Эта версия не выполнила зарегистрированные критерии полного пилота.'
def number(value,digits=9,signed=False):return format(value,('+' if signed else '')+'.'+str(digits)+'f').replace('.',',')
point=result['pooled'];lines=[]
for arm in ['native_B0','legacy_unfiltered_G','old_G7','old_C34',*p['arms']]:
    m=point[arm];lines.append(f"{arm}: HOTA {number(m['HOTA'])}; AssA {number(m['AssA'])}; IDF1 {number(m['IDF1'])}; IDSw {m['IDSw']}; FP {m['CLR_FP']}; FN {m['CLR_FN']}.")
contrasts=[]
for name,c in result['comparisons'].items():contrasts.append(f"{name}: разность HOTA {number(c['delta_pooled_HOTA'],signed=True)}; 99% интервал [{number(float(c['ci99_primary_family'][0]),signed=True)}; {number(float(c['ci99_primary_family'][1]),signed=True)}]; защитные критерии {'выполнены' if c['harm_passed'] else 'не выполнены'}.")
idlines=[]
for arm in p['arms']:
    c=fate['summary'][arm];idlines.append(f"{arm}: разрешены свои {c.get('accepted_own',0)}, чужие {c.get('accepted_foreign',0)}, неизвестные {c.get('accepted_unknown',0)}; отвергнуты свои {c.get('refused_own',0)}, чужие {c.get('refused_foreign',0)}, неизвестные {c.get('refused_unknown',0)}. Общий решатель выбрал свои {c.get('global_selected_own',0)}, чужие {c.get('global_selected_foreign',0)}; публичный старый ID на кандидате: свои {c.get('old_ID_emitted_own',0)}, чужие {c.get('old_ID_emitted_foreign',0)}. Это запросы, повторные наблюдения не являются независимыми объектами.")
geometry_delta=point['C34']['HOTA']-point['G_DOMAIN']['HOTA'];baseline_delta=point['C34']['HOTA']-point['native_B0']['HOTA']
limit='Выборка — 20 ранее изученных проверочных видео; она не является новым независимым финальным тестом. Локальное разрешение не равно публичному старому ID. Прямые метки подтверждают только точные прошлое и текущее наблюдения, не чистоту всей истории. Неизвестные не заполнены. Отсутствие наблюдаемого возврата не доказывает безвозвратную потерю. Ошибка ядра не установлена.'
decision=('Подготовить независимую проверку после разбора прямыхID; эта запись не запускает её автоматически.' if accepted else 'Не переносить эту версию на независимый финальный тест. Разобрать первичные прямые случаи по звеньям: балл, стоимость, общий выбор, дубликаты, публичныйID и дальнейшая судьба; выбрать одну следующую правку по накопленному заключению. Пороги, alpha и точность этого опыта не менять.')
report='# Рабочее обучение пространственного считывателя: полный нативный пилот\n\n1. Итог\n\n'+outcome+' '+limit+'\n\n2. Условия\n\nПять фиксированных вариантов на 20 видео, те же детекции YOLO26 и штатный жизненный цикл треков. Прежнее относительное считывание, 19 сдвигов, float32 (числа одинарной точности), поправка стоимости 0,2, порог сопоставления 0,8. Новые модели обучены на 26 рабочих связях 20 других видео. G_DOMAIN применяет геометрическую поправку в том же допуске без сигнала и обучаемого считывателя. Источники: protocol.json, deployment.json, evaluation_protocol.json. Готовые исходные выходы, метрики и карты переиспользованы.\n\n3. Метрики\n\n'+'\n\n'.join(lines)+'\n\nИсточник: comparison_report.json → pooled.\n\n'+'\n\n'.join(contrasts)+'\n\nИсточник: comparison_report.json → comparisons.\n\n4. Прямые наблюдения ID\n\n'+'\n\n'.join(idlines)+'\n\nИсточник: id_fate.json → summary и records. Конкуренты, удаления дубликатов и последующие реальные размеченные наблюдения сохранены по каждому запросу.\n\n5. Проверки\n\nДва новых полных нулевых свидетеля точно совпали по массивам и размеченным JSON; 40 готовых контролей 20 видео переиспользованы. 58 тестов каталога перед запуском пройдены. Скалярные баллы, геометрия и матрицы проверены без переобучения и повтора трекера: runtime_audit.json. Агрегирование HOTA сверено с официальным TrackEval: comparison_report.json → bootstrap.\n\n6. Решение\n\n'+decision+'\n\n7. Относительное изменение\n\n'+f"FlyVis против исходного трекера: разность HOTA {number(baseline_delta,signed=True)}, относительное изменение {number(100*baseline_delta/point['native_B0']['HOTA'],6,True)}%. Против геометрии в том же допуске: {number(geometry_delta,signed=True)}, {number(100*geometry_delta/point['G_DOMAIN']['HOTA'],6,True)}%. Доля измеренного потолка не определена: потолок в этом опыте не измерялся.\n"
path=OUT/'РЕЗУЛЬТАТ.md';assert not path.exists();path.write_text(report,encoding='utf8')
path=ROOT/'experiments/project_history.json';before=path.read_bytes();catalog=json.loads(before);card=next(c for c in catalog['experiments'] if c['id']=='domain_matched_native_identity_20261109')
assert card['state'] in ['planned','running']
updates=dict(obtained=' '.join(lines)+' Источник: runs/domain_matched_native_20261109/comparison_report.json → pooled. ПрямыеID: runs/domain_matched_native_20261109/id_fate.json → summary,records.',
    interpretation=outcome+' СравнениеC34сG_DOMAINотделяетвкладзрительногосигналаотоднойпоправкистоимости;обычныевходныевариантыотделяютсвойстваFlyVisотпространственно-временнойобработкивхода. Пооднимразрешениямнеприписыватьприростядру.',decision=decision,cannot_claim=limit)
for field,value in updates.items():
    if card.get(field):card.setdefault('revisions',[]).append(dict(date=now.isoformat(),field=field,old=card[field],reason='Завершён фиксированный нативный пилот, добавлены полный результат и границы вывода.'))
    card[field]=value
card.update(state=state,done='2026-11-09')
for i,criterion in enumerate(card['acceptance']):criterion.update(met=accepted if i==1 else True,evidence='runs/domain_matched_native_20261109/'+('comparison_report.json → comparisons' if i==1 else 'runtime_audit.json; id_fate.json; zero_complete.json'))
card['evidence'] += [dict(kind='file',ref='runs/domain_matched_native_20261109/'+name,claim=claim) for name,claim in [('РЕЗУЛЬТАТ.md','Полный итог фиксированной версии'),('comparison_report.json','Официальные метрики и парные интервалы'),('id_fate.json','Прямые свои/чужие и последующая судьбаID'),('runtime_audit.json','Независимая сверка баллов и общего решателя')]]
atomic(path,(json.dumps(catalog,ensure_ascii=False,indent=2)+'\n').encode(),before)
path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';before=path.read_bytes();nl=b'\r\n' if b'\r\n' in before else b'\n';marker='## Журнал изменений'.encode()+nl
body='**'+now.strftime('%d.%m.%Y, %H:%M MSK')+'** — Завершён нативный пилот рабочего обучения пространственных карт\n\n*Получено (`domain_matched_native_identity_20261109`).* '+outcome+' '+' '.join(lines)+' Источник runs/domain_matched_native_20261109/comparison_report.json → pooled,comparisons; полныеIDruns/domain_matched_native_20261109/id_fate.json. Независимаяпроверкабаллов/геометрии/матрицruntime_audit.jsonпройдена;готовыеобучения,карты,исходныйтрекеристарыеметрикинеповторялись.\n\n*Вывод.* '+limit+'\n\n*Решение.* '+decision+'\n\n'
atomic(path,before.replace(marker,marker+nl+body.replace('\n',nl.decode()).encode(),1),before)
write('recording.json',dict(passed=True,recorded=now.isoformat(),native_pilot_accepted=accepted,goal_achieved=False,
    independent_final_test=False,text_review=dict(method_and_object_separated=True,no_neighbor_tautology=True,actual_and_planned_separated=True),sources={name:sha(OUT/name) for name in ['comparison_report.json','id_fate.json','runtime_audit.json','РЕЗУЛЬТАТ.md']}))
print('NATIVE_RESULT_RECORDED',state,'goal remains open',flush=True)
