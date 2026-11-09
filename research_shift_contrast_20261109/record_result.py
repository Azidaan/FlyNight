"""Close only this contrast implementation; preserve primary records and history."""
import json,hashlib,time
from pathlib import Path
from datetime import datetime,timezone,timedelta
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent;NATIVE=ROOT/'runs/domain_matched_native_20261109'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def atomic(path,data,before):
    temp=path.with_suffix(path.suffix+'.contrast_result.pending');assert not temp.exists();temp.write_bytes(data)
    for _ in range(30):
        assert path.read_bytes()==before
        try:temp.replace(path);return
        except PermissionError:time.sleep(.1)
    raise PermissionError(str(path))
now=datetime.now(timezone(timedelta(hours=3)));summary=read(OUT/'summary.json');audit=read(OUT/'audit.json');profiles=read(OUT/'profiles.json');assert summary['passed'] and audit['passed'] and not summary['any_gate']
# Descriptive output on the three predeclared oldval cases, never a fitting gate.
model=read(NATIVE/'deployment.json');theta=np.array(model['models']['C34']['theta']);stats=model['geometry_normalization'];diagnostic=[]
for r in profiles['diagnostic']:
    profile=np.array(r['profile']['profiles']);g=(np.array(r['geometry'])-stats['mean'])/stats['scale'];geometry=float(g@theta[:7]+theta[-1]);component=profile@theta[7:-1];z=geometry+component;winner=r['profile']['winner'];assert abs(float(z[winner])-r['old_logit'])<1e-12
    diagnostic.append(dict(video=r['video'],frame=r['frame'],kind=r['kind'],old_selected_logit=r['old_logit'],all_shift_logit_min=float(z.min()),all_shift_logit_max=float(z.max()),weighted_peak_minus_median=float(component.max()-np.median(component)),weighted_profile_std=float(component.std()),winner=winner,all_shift_logits=z.tolist(),not_used_for_gate=True))
with (OUT/'diagnostic_profiles.json').open('x',encoding='utf8') as f:json.dump(dict(passed=True,records=diagnostic,sources={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),OUT/'profiles.json',NATIVE/'deployment.json']},scope='Only predeclared descriptive cases; no threshold selection or valid universal abort from negative raw scores.'),f,ensure_ascii=False,indent=2)
lines=[]
for arm in ['C34','R134_spatial','R134_temporal']:
    a=summary['metrics'][arm+'_contrast'];b=summary['ready_baselines'][arm];c=summary['metrics'][arm+'_scale_only']
    lines.append(f"{arm}: контраст — свои {a['own_accept']}/20, чужие {a['foreign_accept']}/6; прежний — свои {b['own_accept']}/20, чужие {b['foreign_accept']}/6; только масштаб — свои {c['own_accept']}/20, чужие {c['foreign_accept']}/6.")
text='# Контраст корреляционного пика: отрицательный результат\n\n1. Итог\n\nНи один вариант не прошёл заранее заданный допуск. Полный трекер этой версии не запускается. Это не доказательство бесполезности всех способов оценки формы отклика.\n\n2. Измерено\n\n'+'\n\n'.join(lines)+'\n\nИсточник: summary.json → metrics,ready_baselines,gates. Пороги и штраф не подбирались; прежние модели не переобучались.\n\n3. Граница вывода\n\nВычитание среднего дальних сдвигов при прежнем выборе сдвига ухудшило локальное различение. Контроль масштаба не воспроизвёл это ухудшение FlyVis. Это не устанавливает единственную причину и не доказывает, что любой PSR бесполезен. Те же обычные входные варианты также не прошли допуск. Новые 120 моделей проверены независимо; карты, сеть, видео и исходные трекеры не повторялись. Источники: profiles.json,audit.json.\n\n4. Что дальше\n\nСохранить отрицательный итог и вернуться к проверенным ограничениям сопоставления. Перенос согласованной геометрии ранее проверялся при старых фиксированных весах; совместная проверка с обучением на тех же рабочих26связях отличается от той попытки. Перед новым расчётом требуется отдельная постановка с одинаковыми изображением, геометрией, опорой и обучением.\n'
(OUT/'РЕЗУЛЬТАТ.md').write_text(text,encoding='utf8')
path=ROOT/'experiments/project_history.json';before=path.read_bytes();cat=json.loads(before);card=next(c for c in cat['experiments'] if c['id']=='shift_profile_contrast_20261109');assert card['state']=='planned'
updates=dict(obtained=' '.join(lines)+' Источник: research_shift_contrast_20261109/summary.json → metrics,ready_baselines,gates. Независимо проверены120моделей, источник audit.json.',interpretation='Эта реализация вычитания бокового среднего не прошла ни один допуск. Контроль масштаба сохранил прежний результатC34, поэтому одного масштаба недостаточно для объяснения ухудшения. Карты и сеть не менялись. Из этого не следует бесполезность любой оценки формы пика.',decision='Не запускать нативный пилот этой версии. Проверить согласованную геометрию с одинаковым новым обучением как отдельный кандидат; не повторять прежний фиксированный координатный опыт.',cannot_claim='Это прежние26разработческихсвязей, а не независимый тест или сохранённыеID. Тристарыхvalпримера не влияли на обучение, допуск или пороги. Не установлена ошибкаядраFlyVis; общий принципPSRнеопровергнут.')
for field,value in updates.items():
    if card.get(field):card.setdefault('revisions',[]).append(dict(date=now.isoformat(),field=field,old=card[field],reason='Завершён фиксированный контрастный считыватель с масштабным контролем и независимым аудитом.'))
    card[field]=value
card.update(state='completed_negative',done='2026-11-09')
for i,c in enumerate(card['acceptance']):c.update(met=i!=1,evidence='research_shift_contrast_20261109/'+('summary.json → gates' if i==1 else 'profiles.json; audit.json'))
card['evidence'] += [dict(kind='file',ref='research_shift_contrast_20261109/'+name,claim=claim) for name,claim in [('summary.json','Все фиксированные контроли'),('audit.json','Независимая проверка120моделей'),('РЕЗУЛЬТАТ.md','Отрицательный итог конкретной реализации')]]
atomic(path,(json.dumps(cat,ensure_ascii=False,indent=2)+'\n').encode(),before)
path=ROOT/'experiments/useful_literature.json';before=path.read_bytes();lit=json.loads(before);identity='bolme2010_peak_confidence_scope';assert all(s['id']!=identity for s in lit['sources'])
lit['sources'].append(dict(id=identity,title='Когда качество пика помогает отслеживанию — MOSSE',problem_group='Сохранение личности',method_group='Контраст пространственного совпадения',publication='David S.Bolme,J.Ross Beveridge,Bruce A.Draper,Yui Man Lui. Visual Object Tracking using Adaptive Correlation Filters. CVPR2010; просмотрена расширенная авторская версия.',url='https://www.cs.colostate.edu/~draper/papers/bolme_cvpr10.pdf',doi='IEEEномер5539960;DOIнепровереннастраницеиз-запроверкиJavaScript',sections='Раздел3.5иобсуждениепростогофильтрав4.1.',established='PSR использует высоту пика, среднее и разброс боковых откликов. Для простого шаблонного фильтра авторы отмечают неинформативность показателя.',problem='Высокая пространственная корреляция может соответствовать чужому объекту.',application='Источник помог отделить собственноPSRотвычитаниясреднего и потребовать опыт с контролем обычного входа. Наша поканальная19сдвиговаяадаптацияпроверенаиполучилаотрицательныйрезультат.',limitation='Статья не доказывает пользу этого преобразования на картахFlyVis. Отрицательный итог нашей реализации не опровергает MOSSEиливсе показателиформыпика.',evidence=['research_shift_contrast_20261109/REVIEW.md','research_shift_contrast_20261109/summary.json','research_shift_contrast_20261109/audit.json'],related=['sys:С2']))
atomic(path,(json.dumps(lit,ensure_ascii=False,indent=2)+'\n').encode(),before)
path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';before=path.read_bytes();nl=b'\r\n' if b'\r\n' in before else b'\n';marker='## Журнал изменений'.encode()+nl
entry='**'+now.strftime('%d.%m.%Y, %H:%M MSK')+'** — Контраст профиля сдвигов не прошёл допуск\n\n*Получено (`shift_profile_contrast_20261109`).* '+' '.join(lines)+' Источник research_shift_contrast_20261109/summary.json. Все120новыхмоделейпроверены,audit.json; прежниемодели,карты,видеоиетьнеповторялись. Тристарыхпримераописаныотдельнобезвлияниянадопуск:diagnostic_profiles.json.\n\n*Вывод.* Вычитаниебоковогосреднегоприсохранённомвыборесдвигаухудшилочтение. КонтрольмасштабаC34сохранилпрежнийрезультат;общуюпричинупровалаядраэтонеустанавливает. ПолезныйпервичныйMOSSEисточниксохранёнвреестре;различиеоригинальногоPSRинашейадаптацииуказано.\n\n*Решение.* ПолныйтрекерэтойверсииНЕзапускать. Следующийотдельныйкандидат—согласованнаягеометриясодинаковымобучениемивидимойопорой;прежнийкоординатныйопытпристарыхвесахнеповторять.\n\n'
atomic(path,before.replace(marker,marker+nl+entry.replace('\n',nl.decode()).encode(),1),before)
with (OUT/'recording.json').open('x',encoding='utf8') as f:json.dump(dict(passed=True,state='completed_negative',no_native_pilot=True,goal_achieved=False,sources={n:sha(OUT/n) for n in ['summary.json','audit.json','РЕЗУЛЬТАТ.md','diagnostic_profiles.json']}),f,ensure_ascii=False,indent=2)
print('CONTRAST_NEGATIVE_RECORDED_NO_NATIVE_PILOT',flush=True)
