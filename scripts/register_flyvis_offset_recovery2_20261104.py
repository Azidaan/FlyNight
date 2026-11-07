"""До расчёта фиксирует выборку, причинные контроли и критерий для С7."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib,itertools,json,os,shutil,sys
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/flyvis_offset_recovery2_20261104'
sys.path.insert(0,str(ROOT))
from scripts.flyvis_offset_recovery2_20261104 import positions,frames_for
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 OUT.mkdir(exist_ok=False)
 original=json.loads((ROOT/'runs/flyvis_tau_half_20261104/protocol.json').read_text(encoding='utf-8'))
 old=set(original['videos']);selection=json.loads((ROOT/'runs/train_cache/selection.json').read_text(encoding='utf-8'))['videos']
 take=lambda split:[x for x in selection[split] if x not in old and (ROOT/'data/bdd100k/videos/train'/(x+'.mov')).exists()][:4]
 train,test=take('train'),take('dev');assert len(train)==len(test)==4 and not set(train)&set(test)
 cases=[]
 for scene,contrast,speed,axis,start in itertools.product(range(8),[1.,.25],[1,4],[0,1],['still','moving']):
  cases.append(dict(id=f's{scene}_c{contrast}_v{speed}_a{axis}_{start}',scene=scene,split='calibration' if scene<4 else 'evaluation',contrast=contrast,speed=speed,axis=axis,start=start,event='reverse'))
 assert len(cases)==128
 shutil.copy2(ROOT/'runs/flyvis_tau_half_20261104/check_catalog.py',OUT/'check_catalog.py')
 p=dict(original);p.update(id='flyvis_offset_recovery2_v1',registered_msk=datetime.now(ZoneInfo('Europe/Moscow')).isoformat(),videos=train+test,cases=cases,
  dataset='Первые четыре доступных TRAIN и первые четыре DEV из фиксированного train_cache/selection.json, исключая четыре ранее изученных видео. Все видео исходного BDD TRAIN; DEV проекта ранее мог участвовать в других работах. На каждом используется первый кадр, глобальное искусственное движение.',
  split='Первые 4 изображения только для коэффициентов; другие 4 для оценки; без выбора по результатам.',
  starts='still: кадры1:30 неподвижны, 31:54 движение; moving: движение с кадра1; для speed4 первые четыре движения имеют скорость2, далее4; в обоих обратное движение с кадра55. Алгоритму не передаётся признак наличия неподвижного начала.',
  geometry='Фрагмент512x512 из исходного кадра720x1280 без дополнения. Для скорости4 и вертикальной оси исходный фрагмент прижат к нижней границе одинаково для обоих начал; иначе центр. В условии moving/speed4 первые четыре перехода равны2px, затем4px: это позволяет оставаться внутри кадра; к периоду11:20 скорость уже неизменна.',
  recovery='Предыдущая recovery1 завершилась с PermissionError при атомарной замене status.json после 69 законченных случаев штатной модели. Все научные условия неизменны, добавлены ограниченные повторы служебной записи; 128 входов и 69 завершённых выходов повторно используются строго по SHA.',
  previous_protocol_sha256=sha(ROOT/'runs/flyvis_offset_recovery1_20261104/protocol.json'),previous_controller_sha256=sha(ROOT/'runs/flyvis_offset_recovery1_20261104/chain_result.json'),
  method='Усреднение выходного 2D поля по центральной маске. Смещение оценивается только по прошедшим кадрам11:20 и фиксируется с кадра21; без предположения в предикторе, что там на самом деле покой. Для каждой версии и метода отдельный положительный скаляр МНК подбирается только на 4 calibration-сценах, кадры21:78, оба контраста/скорости/оси/начала; целевые векторы не подаются модели.',
  controls='Штатная и ускоренная tau*0.5; каждая без/с вычитанием и без/с отдельной калибровкой, всего 8 вариантов выходного преобразования; нулевой вектор. Одни и те же изображения; никакого переобучения нейросети.',
  metric='На 4 evaluation-сценах по 16 условий на каждый контраст и каждое начало: средняя векторная ошибка по кадрам21:78; ошибка до/после разворота; задержка первого трёхкратного правильного знака начиная с кадра55 (случаи без перехода указываются явно); доля правильного знака до разворота, отклик на начальных доступных кадрах.',
  primary='Предварительный составной критерий для всех четырёх сочетаний контраста и начала: калиброванная ускоренная с вычитанием не хуже по средней ошибке, чем калиброванная ускоренная без вычитания и чем калиброванная штатная с вычитанием; знаковая задержка определена на всех случаях и в среднем не хуже штатной. Независимая математическая проверка обязательна.',
  limits='Четыре изображения на каждую группу — исследовательская проверка, не независимый итоговый тест; искусственный глобальный сдвиг, один экземпляр FlyVis, известный только на стадии оценки. Не HOTA, не объекты и не гарантия применимости к естественному видео. Подбор различает сцены, но модель и данные из других опытов не являются полностью независимыми.',
  hard_stop='2026-11-05T07:39:30+03:00',save_by='2026-11-05T07:30:00+03:00',fps=30.,dt=1/60,frames=79,training=False,
  parameter_intervention='Две версии: неизменённая и предварительно заданные nodes_time_const*0.5; веса на диске не меняются.',
  audit='Все хеши выходов и входов; шесть независимо восстановленных кадров каждого случая; независимый пересчёт коэффициентов, ошибок и знаковых задержек; полная повторная симуляция FlyVis не заявляется.')
 for name in ['scripts/flyvis_offset_recovery2_20261104.py','scripts/analyze_flyvis_offset_recovery2_20261104.py','scripts/audit_flyvis_offset_recovery2_20261104.py','scripts/control_flyvis_offset_recovery2_20261104.py','scripts/guard_flyvis_offset_recovery2_20261104.py','runs/flyvis_offset_recovery2_20261104/check_catalog.py']:
  p['frozen_sources'][name]=sha(ROOT/name)
 (OUT/'protocol.json').write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 (OUT/'authorization.json').write_text(json.dumps(dict(message='Давай проверять',scope='Сравнение вычитания фонового смещения с контролями и новыми изображениями',protocol_sha256=sha(OUT/'protocol.json')),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 prior=ROOT/'runs/flyvis_offset_recovery1_20261104';before=json.loads((prior/'protocol.json').read_text(encoding='utf-8'))
 for key in ['videos','cases','starts','geometry','method','controls','metric','primary','fps','dt','frames']:
  assert p[key]==before[key],key
 (OUT/'inputs').mkdir();reuse=[];old_dir=prior/'inputs'
 for case in cases:
  source=old_dir/(case['id']+'.npz');mark=source.with_suffix('.json');assert sha(source)==json.loads(mark.read_text())['sha256']
  for f in [source,mark]:os.link(f,OUT/'inputs'/f.name)
  reuse.append(dict(id=case['id'],sha256=sha(source)))
 (OUT/'reused_inputs.json').write_text(json.dumps(dict(cases=reuse,old_protocol_sha256=p['previous_protocol_sha256']),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 (OUT/'stock'/'outputs').mkdir(parents=True);saved=[];frozen=json.loads((prior/'stock'/'model_loaded.json').read_text())['frozen_parameter_sha']
 for case in cases:
  src=prior/'stock'/'outputs'/(case['id']+'.npz');marker=src.with_suffix('.json')
  if not marker.exists():continue
  m=json.loads(marker.read_text());assert sha(src)==m['sha256'] and m['protocol_sha256']==p['previous_protocol_sha256']
  assert m['input_sha256']==sha(OUT/'inputs'/(case['id']+'.npz')) and m['parameter_sha256']==frozen
  target=OUT/'stock'/'outputs'/src.name;os.link(src,target)
  m.update(original_protocol_sha256=m['protocol_sha256'],original_marker_sha256=sha(marker),protocol_sha256=sha(OUT/'protocol.json'),reused_from=str(src.relative_to(ROOT)).replace('\\','/'))
  target.with_suffix('.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');saved.append(dict(id=case['id'],sha256=m['sha256'],old_marker_sha256=m['original_marker_sha256']))
 assert len(saved)==69
 (OUT/'reused_stock_outputs.json').write_text(json.dumps(dict(cases=saved,previous_protocol_sha256=p['previous_protocol_sha256']),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 catalog=ROOT/'experiments/project_history.json';shutil.copy2(catalog,OUT/'project_history_before.json');data=json.loads(catalog.read_text(encoding='utf-8'));assert not any(x['id']==p['id'] for x in data['experiments'])
 former=next(x for x in data['experiments'] if x['id']=='flyvis_offset_recovery1_v1');assert former['state']=='planned'
 former.update(state='infrastructure',done='2026-11-04',obtained='Подготовлены 128 входов и 69 полных выходов штатной FlyVis; следующая служебная запись прервалась с PermissionError. Источник: runs/flyvis_offset_recovery1_20261104/chain_result.json и stock_stderr.log.',interpretation='Технический отказ атомарной замены состояния; научное сравнение не завершено.',decision='Продолжить отдельной recovery2 с ограниченными повторами записи и повторным использованием готовых данных по SHA.',cannot_claim='Это не полный результат сравнения и не HOTA.')
 data['experiments'].append(dict(id=p['id'],title='FlyVis — Проверка вычитания фонового смещения при движении',area='flyvis',state='planned',started='2026-11-04',tags=['Y1','motion-diagnostic'],question='Уменьшает ли причинное вычитание исходного смещения ошибку ускоренной FlyVis без потери медленного движения и быстрой реакции?',hypothesis='Предварительное смещение искажает масштаб, но вычитание может подавить настоящее движение, если последовательность начинается в движении.',expected='Получить парное сравнение двух версий на ранее не использованных в этом подборе изображениях, с честным контролем отсутствия неподвижного начала.',acceptance=[dict(criterion=p['primary'],met=None)],done='',obtained='',interpretation='',decision='',cannot_claim='',metrics=[],evidence=[dict(kind='json',ref='runs/flyvis_offset_recovery2_20261104/protocol.json',claim='Данные, условия, сравнения и критерий до расчёта')]))
 catalog.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 plan=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';raw=plan.read_bytes();mark='## Журнал изменений'.encode();nl=b'\r\n' if mark+b'\r\n' in raw else b'\n';now=datetime.now(ZoneInfo('Europe/Moscow'))
 e=f'\n**{now:%d.%m.%Y, %H:%M} MSK** — Продолжение после блокировки файла статуса FlyVis\n\n`flyvis_offset_recovery1_v1` завершилась с PermissionError при замене `stock/status.json`; код1 и исходный контроллер сохранены `runs/flyvis_offset_recovery1_20261104/chain_result.json`, фактическое окончание PID проверено. Отдельная `flyvis_offset_recovery2_v1` ограниченно повторяет служебную запись и без изменений научных условий повторно использует все 128 входов и 69 завершённых штатных выходов по SHA: `runs/flyvis_offset_recovery2_20261104/reused_inputs.json`, `reused_stock_outputs.json`. Неполный случай не является готовым результатом. Протокол `runs/flyvis_offset_recovery2_20261104/protocol.json`; не HOTA.\n\n';assert raw.count(mark+nl)==1;plan.write_bytes(raw.replace(mark+nl,mark+nl+e.encode(),1))
 (OUT/'status.json').write_text(json.dumps(dict(stage='registered',done=0,total=128))+'\n');print('registered')
if __name__=='__main__':main()
