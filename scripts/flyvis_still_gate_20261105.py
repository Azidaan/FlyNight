"""Проспективный дешёвый пилот: причинный допуск вычитания по текущему изображению."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse,hashlib,json,sys
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/flyvis_still_gate_20261105';SRC=ROOT/'runs/flyvis_offset_recovery2_20261104'
sys.path.insert(0,str(ROOT))
from scripts.flyvis_offset_recovery2_20261104 import positions
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def stationary(frames):
 # Используются исключительно кадры10–20, известные до первого решения в кадре21.
 return all(np.array_equal(frames[j],frames[j-1]) for j in range(11,21))
def register():
 OUT.mkdir(exist_ok=False)
 old=json.loads((SRC/'protocol.json').read_text(encoding='utf-8'));a=json.loads((SRC/'independent_audit.json').read_text(encoding='utf-8'));assert a['passed']
 prior=json.loads((SRC/'report.json').read_text(encoding='utf-8'));assert prior['complete']
 # Пробные пять вызовов phaseCorrelate с одной уже исследованной evaluation-сцены выполнены до регистрации только для проверки направления координат.
 info=dict(id='flyvis_still_gate_v1',registered_msk=datetime.now(ZoneInfo('Europe/Moscow')).isoformat(),
   question='Не допускать вычитания смещения, если в начальных кадрах есть наблюдаемое движение.',
   inspiration='Wallflower (Toyama et al., 1999) связывает обновление модели фона с предсказуемостью, но наш признак равенства кадров не является его алгоритмом: https://www.microsoft.com/en-us/research/publication/wallflower-principles-and-practice-of-background-maintenance/',
   prior_report_sha256=sha(SRC/'report.json'),prior_protocol_sha256=sha(SRC/'protocol.json'),prior_audit_sha256=sha(SRC/'independent_audit.json'),script_sha256=sha(Path(__file__)),audit_script_sha256=sha(ROOT/'scripts/audit_flyvis_still_gate_20261105.py'),
   validation_caveat='Четыре сцены calibration и четыре evaluation использованы в предыдущем опыте; исследовательский повтор, не новая независимая проверка. До регистрации технически просматривались значения phaseCorrelate на одном evaluation-видео, кадры 15,21,54,55,56; результат этих кадров нельзя считать ослеплённым.',
   pre_registration_probe=dict(case='s4_c1.0_v1_a0_moving',frames=[15,21,54,55,56],shifts=[[1.0015728115,-.0007040253],[1.0105345183,.0017608472],[.9960680915,.00112214496],[-.9960680766,-.0011221430],[-1.0016324780,-.0031997126]],source_sha256=sha(SRC/'inputs/s4_c1.0_v1_a0_moving.npz'),purpose='Техническая проверка направления координат; пять оценочных кадров просмотрены до регистрации.'),
   gate='На момент t=21 сравнить изображения t=10..20: только если все 10 переходов побайтно совпадают, вычитать среднее выхода FlyVis t=11..20, иначе не вычитать. Решение и оценка используют исключительно прошлые кадры, не зависят от GT; неизменность пикселей не доказывает отсутствия физического движения.',
   variants='Штатная и tau×0.5: без вычитания; безусловное среднее11..20; условное вычитание с отказом в неопределённости; вычитание отклика после штатного прогрева на первом кадре (уже существующее правило FlyVis). Для первых двух используются коэффициенты МНК предыдущего опыта; условное правило выбирает коэффициент соответствующей ветви; для первого кадра коэффициент обучается на calibration. Физический baseline — причинная фазовая корреляция двух кадров и нулевой поток, без FlyVis.',
   acceptance='На evaluation раздельно для still/moving и контраста1/.25: ошибка условного варианта не хуже неизменённого и безусловного вычитания при обоих стартах; любые ложные допуски на известных moving показываются явно. Результат только исследовательский и не является критерием закрытия С7.',
   stress='Отдельно только для классификатора входных кадров: изменить один пиксель в одном начальном кадре неподвижных случаев и измерить долю отказов; сравнить две одинаковые последовательности однородного изображения с разными гипотезами движения. Это не повтор FlyVis на шумных изображениях и не оценка устойчивости полной системы.',
   sources=['https://www.microsoft.com/en-us/research/publication/wallflower-principles-and-practice-of-background-maintenance/','https://www.telecom.uliege.be/publi/publications/barnich/Barnich2011ViBe/index.html','https://www.nature.com/articles/nn858'],
   baseline_notes='Прогрев FlyVis на первом кадре 1с уже реализован в tracking/flyvis_motion.py; вычитание декодированного отклика q0 соответствует штатной опции subtract_initial_baseline=True с прежним входом.',
   accuracy='EPE среднего по центральной маске вектора, кадры21..78; отдельно правильный знак во время движения31..54 и распределение определения задержки знака на 55..78; одинаковые входы и известные сдвиги для оценки.',
   limitation='Точные неподвижные синтетические кадры без моделирования сенсорного шума, глобальный сдвиг, одна модель FlyVis; физическая неподвижность не равна совпадению кадров. Новый признак использует исходные кадры, а не только FlyVis. HOTA, реальные объекты и общая польза ядра не измерены.')
 save(OUT/'protocol.json',info)
 catalog=ROOT/'experiments/project_history.json';data=json.loads(catalog.read_text(encoding='utf-8'));assert not any(c['id']==info['id'] for c in data['experiments'])
 data['experiments'].append(dict(id=info['id'],title='FlyVis — Когда разрешать вычитание фонового смещения',area='flyvis',state='planned',started=datetime.now(ZoneInfo('Europe/Moscow')).date().isoformat(),tags=['Y1','motion-diagnostic'],
   question='Можно ли уменьшить фоновое смещение, не вычитая наблюдаемое движение в начале?',hypothesis='Консервативная проверка идентичности исходных кадров предотвратит безусловное вычитание при заметном движении, но пропустит коррекцию в шуме и не отличит движение однотонной сцены.',expected='Оценить разрешение и отказ на ранее изученной синтетической серии без повторного запуска сети.',acceptance=[dict(criterion=info['acceptance'],met=None)],done='',obtained='',interpretation='',decision='',cannot_claim='',metrics=[],evidence=[dict(kind='json',ref='runs/flyvis_still_gate_20261105/protocol.json',claim='Условия и ограничения до расчёта')]))
 (OUT/'project_history_before.json').write_text(catalog.read_text(encoding='utf-8'),encoding='utf-8');catalog.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 plan=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';raw=plan.read_bytes();marker='## Журнал изменений'.encode();nl=b'\r\n' if marker+b'\r\n' in raw else b'\n';now=datetime.now(ZoneInfo('Europe/Moscow'))
 entry=f'\n**{now:%d.%m.%Y, %H:%M} MSK** — Зарегистрирован контроль условия вычитания смещения FlyVis\n\nПо прямому поручению автора, после анализа присланной литературы, до расчёта зарегистрирован `flyvis_still_gate_v1`: решение по совпадению кадров10–20; при неопределённости использовать выход без вычитания. Сравнить безусловное вычитание, штатный прогрев и вычитание первого кадра, нулевой поток и фазовую корреляцию без FlyVis. Данные и выходы прошлого опыта переиспользуются; сцены evaluation уже изучены, а до регистрации проверен знак фазовой корреляции на пяти кадрах одной такой сцены. Это исследовательская проба, не независимая приёмка/HOTA. Протокол `runs/flyvis_still_gate_20261105/protocol.json`.\n\n';assert raw.count(marker+nl)==1;plan.write_bytes(raw.replace(marker+nl,marker+nl+entry.encode(),1))
 print('registered')

def run():
 p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'));assert p['script_sha256']==sha(Path(__file__)) and p['prior_report_sha256']==sha(SRC/'report.json') and p['prior_protocol_sha256']==sha(SRC/'protocol.json')
 original=json.loads((SRC/'protocol.json').read_text(encoding='utf-8'));prior=json.loads((SRC/'report.json').read_text(encoding='utf-8'));mask=np.load(SRC/'geometry.npz')['mask'];items=[];cv2.setNumThreads(2)
 for c in original['cases']:
  inp=SRC/'inputs'/(c['id']+'.npz');im=json.loads(inp.with_suffix('.json').read_text());assert sha(inp)==im['sha256'];frames=np.load(inp)['frames'];v={}
  for arm in ('stock','fast'):
   f=SRC/arm/'outputs'/(c['id']+'.npz');m=json.loads(f.with_suffix('.json').read_text());assert sha(f)==m['sha256'] and m['input_sha256']==im['sha256'];q=np.load(f)['flow'][0,1][:,mask].mean(1)
   v[arm+'_raw']=q;v[arm+'_mean']=q-q[11:21].mean(0);v[arm+'_first']=q-q[0]
  items.append(dict(case=c,input_path=inp,values=v,truth=positions(c)[1].astype(float),gate=stationary(frames)))
 gains={}
 for arm in ('stock','fast'):
  for mode in ('raw','mean'):gains[arm+'_'+mode]=prior['gains'][arm+('_raw' if mode=='raw' else '_subtract')]
  X=np.concatenate([x['values'][arm+'_first'][21:79].ravel() for x in items if x['case']['split']=='calibration']);Y=np.concatenate([x['truth'][21:79].ravel() for x in items if x['case']['split']=='calibration']);assert float(X@X)>0;g=float((X@Y)/(X@X));assert g>0;gains[arm+'_first']=g
 rows=[]
 for x in items:
  c=x['case'];truth=x['truth'];result={}
  for arm in ('stock','fast'):
   for mode in ('raw','mean','first','gated'):
    pick=('mean' if x['gate'] else 'raw') if mode=='gated' else mode
    q=x['values'][arm+'_'+pick]*gains[arm+'_'+pick];d=q[21:79]-truth[21:79];u=q[31:55,c['axis']];after=q[55:79,c['axis']]<0
    result[arm+'_'+mode]=dict(epe=float(np.linalg.norm(d,axis=1).mean()),before_epe=float(np.linalg.norm(q[31:55]-truth[31:55],axis=1).mean()),positive_fraction=float(np.mean(u>0)),latency=next((i for i in range(22) if after[i:i+3].all()),None))
  result['zero']=dict(epe=float(np.linalg.norm(truth[21:79],axis=1).mean()))
  if c['split']=='evaluation':
   f=np.load(x['input_path'])['frames'];phase=[];invalid=0
   for j in range(21,79):
    a=f[j-1].astype(np.float32);b=f[j].astype(np.float32)
    if np.ptp(a)==0 or np.ptp(b)==0:phase.append((0.,0.));invalid+=1;continue
    z,_=cv2.phaseCorrelate(a,b)
    if not np.isfinite(z).all():phase.append((0.,0.));invalid+=1
    else:phase.append(z)
   result['phase']=dict(epe=float(np.linalg.norm(np.array(phase)-truth[21:79],axis=1).mean()),invalid=invalid)
  rows.append(dict(case=c,gate=x['gate'],results=result))
 groups={}
 for split in ('calibration','evaluation'):
  for start in ('still','moving'):
   for contrast in (1.,.25):
    cohort=[x for x in rows if x['case']['split']==split and x['case']['start']==start and x['case']['contrast']==contrast];assert len(cohort)==16
    summary={}
    for arm in cohort[0]['results']:
     vals=[x['results'][arm] for x in cohort];summary[arm]={'epe':float(np.mean([y['epe'] for y in vals]))}
     if arm not in ('zero','phase'):
      summary[arm]['positive_fraction']=float(np.mean([y['positive_fraction'] for y in vals]));lat=[y['latency'] for y in vals if y['latency'] is not None];summary[arm]['latency_identified']=len(lat);summary[arm]['latency_mean']=float(np.mean(lat)) if len(lat)==16 else None
     if arm=='phase':summary[arm]['invalid_pairs']=sum(y['invalid'] for y in vals)
    summary['gate_true']=sum(x['gate'] for x in cohort);groups[f'{split}_{start}_{contrast}']=summary
 checks={}
 for start in ('still','moving'):
  for ct in (1.,.25):
   g=groups[f'evaluation_{start}_{ct}'];a=g['fast_gated']['epe'];checks[f'{start}_{ct}_no_worse_raw']=a<=g['fast_raw']['epe'];checks[f'{start}_{ct}_no_worse_mean']=a<=g['fast_mean']['epe']
 # Искажение входа только для признака неподвижности: не заявлять испытание всей FlyVis с шумом.
 static=[x for x in items if x['case']['split']=='evaluation' and x['case']['start']=='still'];stress_true=0
 for x in static:
  sample=np.load(x['input_path'])['frames'][10:21].copy();sample[5,0,0]^=np.uint8(1);stress_true+=all(np.array_equal(sample[j],sample[j-1]) for j in range(1,len(sample)))
 flat=np.full((11,8,8),127,np.uint8);same=np.array_equal(flat,flat.copy())
 report=dict(complete=True,exploratory_gate=all(checks.values()),checks=checks,gains=gains,groups=groups,rows=rows,protocol_sha256=sha(OUT/'protocol.json'),
  gate_stress_only=dict(stationary_cases=len(static),single_pixel_perturbation_updates=stress_true,uniform_still_and_motion_indistinguishable=same),
  limitation=p['limitation'])
 save(OUT/'report.json',report);print(json.dumps(dict(checks=checks,gains=gains,groups={k:v for k,v in groups.items() if k.startswith('evaluation')},stress=report['gate_stress_only']),ensure_ascii=False))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['register','run']);args=ap.parse_args();{'register':register,'run':run}[args.stage]()
