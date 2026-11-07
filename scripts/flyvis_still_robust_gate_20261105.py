"""Причинный технический пилот признака покоя; FlyVis не запускается."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse,hashlib,itertools,json,sys,time
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/flyvis_still_robust_gate_20261105';sys.path.insert(0,str(ROOT))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sample(gray,c):
 h,w=gray.shape;cx=w/2;cy=h/2;rng=np.random.default_rng(c['seed']);result=[]
 for t in range(11):
  step=t if c['trajectory']=='constant' else (t if t<=5 else 10-t)
  shift=step*c['speed'];x=cx-(shift if c['axis']==0 else 0);y=cy-(shift if c['axis']==1 else 0)
  crop=cv2.getRectSubPix(gray,(512,512),(x,y)).astype(np.float32)
  crop=127.5+c['contrast']*(crop-127.5)
  if c['noise_sigma']>0:crop+=rng.normal(0,c['noise_sigma'],crop.shape).astype(np.float32)
  result.append(np.rint(np.clip(crop,0,255)).astype(np.uint8))
 return np.stack(result)
def extract(frames,quality_floor,texture_floor):
 exact=all(np.array_equal(frames[i],frames[i-1]) for i in range(1,11));shifts=[];quality=[];texture=[]
 for i in range(1,11):
  a=frames[i-1].astype(np.float32);b=frames[i].astype(np.float32);texture.append(min(float(a.std()),float(b.std())))
  shift,response=cv2.phaseCorrelate(a,b);shifts.append(float(np.hypot(*shift)));quality.append(float(response))
 assert np.isfinite(shifts).all() and np.isfinite(quality).all()
 observable=bool(min(quality)>=quality_floor and min(texture)>texture_floor)
 return dict(exact=exact,maximum_displacement=max(shifts),minimum_response=min(quality),minimum_texture=min(texture),observable=observable)
def register():
 OUT.mkdir(exist_ok=False);sel=json.loads((ROOT/'runs/train_cache/selection.json').read_text(encoding='utf-8'))['videos'];used=set(json.loads((ROOT/'runs/flyvis_offset_recovery2_20261104/protocol.json').read_text(encoding='utf-8'))['videos'])|set(json.loads((ROOT/'runs/flyvis_tau_half_20261104/protocol.json').read_text(encoding='utf-8'))['videos'])
 take=lambda split:[x for x in sel[split] if x not in used and (ROOT/'data/bdd100k/videos/train'/(x+'.mov')).exists()][:4]
 train,test=take('train'),take('dev');assert len(train)==len(test)==4 and not set(train)&set(test)
 scenarios=[('constant',0.),('constant',.25),('constant',1.),('oscillatory',.25),('oscillatory',1.)];cases=[]
 for scene,texture,contrast,noise,axis,(trajectory,speed) in itertools.product(range(8),('natural','flat'),(1.,.25),(0.,1.),(0,1),scenarios):
  key=f's{scene}_{texture}_c{contrast}_n{noise}_a{axis}_{trajectory}_v{speed}';seed=int(hashlib.sha256(key.encode()).hexdigest()[:16],16)
  cases.append(dict(id=key,scene=scene,split='calibration' if scene<4 else 'evaluation',texture=texture,contrast=contrast,noise_sigma=noise,axis=axis,trajectory=trajectory,speed=speed,seed=seed))
 assert len(cases)==640
 info=dict(id='flyvis_robust_still_gate_v1',registered_msk=datetime.now(ZoneInfo('Europe/Moscow')).isoformat(),videos=train+test,cases=cases,
  purpose='До FlyVis проверить, возможно ли безопаснее узнавать покой в начальных кадрах при шуме, слабом контрасте и малом либо возвращающемся движении.',
  split='Первые 4 ранее не использованные в двух диагностических сериях изображения — calibration, ещё 4 — evaluation; все исходные BDD TRAIN. Прошлые исследования проекта могли использовать эти видео; не итоговый независимый тест.',
  data='Первый кадр каждого видео; 512x512 фрагмент с причинной последовательностью11 кадров. Условия: обычный и ослабленный контраст; текстурная сцена или однотонная; шум отсутствует или независимый аддитивный гауссовский sigma1 уровень8бит на кадр; оси x/y; покой, 0.25 или1px/кадр постоянного движения либо возврат в начальную точку к кадру10. Нет будущих кадров или разметки на входе правила.',
  candidate='По каждому из десяти последовательных переходов t=0..10 вычислить phaseCorrelate в 2D; взять максимальный модуль смещения. Если для любого перехода отклик ниже 0.2 или std одного из кадров <=1 уровень8бит — неопределённо и коррекция запрещена. Эти пороги фиксированы до измерений как инженерные, не получены из статьи. Иначе: покой, только если максимум меньше выученного порога; движение — иначе. Разметка не передаётся предиктору.',
  fit='Только calibration и наблюдаемые движущиеся случаи: threshold = 0.5 * min(maximum_displacement); если таких случаев нет или минимум равен нулю — threshold=0 и обновление запрещено. Это консервативный эмпирический порог, не гарантия истинного покоя.',
  baselines='Безусловный отказ от обновления; точное совпадение всех 10 переходов из предыдущей пробы. Все варианты по изображениям, без FlyVis.',
  acceptance='На evaluation: среди случаев движения доля ошибочно разрешивших обновление <=5% отдельно для каждой комбинации natural/flat, контраста и шума; среди естественных сцен с покоем доля разрешивших обновление >=50% отдельно для каждой комбинации контраста и шума. Пропуск покоя — потеря возможности коррекции, не ложная неподвижность. Требуется независимый пересчёт. Малые выборки, результат предварительный.',
  caveat='Это проверка признака по кадрам, не полный трекер и не эксперимент о качестве FlyVis. Движение всей картинки синтетическое; локальные объекты и реальные помехи не воспроизведены. Гипотеза глобального движения, меняющееся освещение и потеря текстуры способны сломать правило. Положительный результат не позволяет заявлять HOTA.',
  source='Условное обновление фона мотивировано Wallflower, Toyama и др. (1999), но алгоритм фазовой корреляции и пороги здесь являются отдельной упрощённой проверкой: https://www.microsoft.com/en-us/research/publication/wallflower-principles-and-practice-of-background-maintenance/',
  script_sha256=sha(Path(__file__)),audit_sha256=sha(ROOT/'scripts/audit_flyvis_still_robust_gate_20261105.py'),previous_report_sha256=sha(ROOT/'runs/flyvis_still_gate_20261105/report.json'),opencv=cv2.__version__,numpy=np.__version__)
 save(OUT/'protocol.json',info)
 cat=ROOT/'experiments/project_history.json';d=json.loads(cat.read_text(encoding='utf-8'));assert not any(x['id']==info['id'] for x in d['experiments']);(OUT/'project_history_before.json').write_text(cat.read_text(encoding='utf-8'),encoding='utf-8')
 d['experiments'].append(dict(id=info['id'],title='FlyVis — Надёжность признака неподвижности при слабом движении и шуме',area='flyvis',state='planned',started=datetime.now(ZoneInfo('Europe/Moscow')).date().isoformat(),tags=['Y1','motion-diagnostic'],question='Можно ли безопаснее допускать вычитание фона при неизвестном начале движения?',hypothesis='Учёт наблюдаемости и истории с фазовой корреляцией может сохранить часть неподвижных сцен при шуме, но слабая текстура и локальное движение ограничивают распознавание.',expected='Технический контроль точности признака до внедрения в FlyVis.',acceptance=[dict(criterion=info['acceptance'],met=None)],done='',obtained='',interpretation='',decision='',cannot_claim='',metrics=[],evidence=[dict(kind='json',ref='runs/flyvis_still_robust_gate_20261105/protocol.json',claim='Условия и ограничения до расчёта')]))
 cat.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 plan=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';raw=plan.read_bytes();marker='## Журнал изменений'.encode();nl=b'\r\n' if marker+b'\r\n' in raw else b'\n';now=datetime.now(ZoneInfo('Europe/Moscow'));entry=f'\n**{now:%d.%m.%Y, %H:%M} MSK** — Зарегистрирована проверка устойчивого признака неподвижности\n\nПосле присланной литературы зафиксирован дешёвый пилот `flyvis_robust_still_gate_v1`: 4 новые calibration и 4 evaluation-сцены; искусственный сдвиг0/0,25/1px, возврат, шум sigma1, два контраста и однотонный контроль. Порог учится только на calibration; неоднозначность запрещает обновление. Простые контроли — точное совпадение кадров и отказ от обновления. Это проверка признака на входных кадрах, не запуск FlyVis и не HOTA. Протокол `runs/flyvis_still_robust_gate_20261105/protocol.json`.\n\n';assert raw.count(marker+nl)==1;plan.write_bytes(raw.replace(marker+nl,marker+nl+entry.encode(),1));print('registered')
def run():
 p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'));assert p['script_sha256']==sha(Path(__file__)) and p['audit_sha256']==sha(ROOT/'scripts/audit_flyvis_still_robust_gate_20261105.py');cv2.setNumThreads(2)
 images={};sources=[]
 for i,vid in enumerate(p['videos']):
  path=ROOT/'data/bdd100k/videos/train'/(vid+'.mov');cap=cv2.VideoCapture(str(path));ok,img=cap.read();cap.release();assert ok;images[i]=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY);sources.append(dict(video=vid,sha256=sha(path),frame_sha256=hashlib.sha256(images[i].tobytes()).hexdigest()))
 rows=[];start=time.perf_counter()
 for c in p['cases']:
  gray=images[c['scene']] if c['texture']=='natural' else np.full_like(images[c['scene']],127);frames=sample(gray,c);feat=extract(frames,.2,1.)
  rows.append(dict(case=c,features=feat,frames_sha256=hashlib.sha256(frames.tobytes()).hexdigest()))
  if len(rows)%40==0:save(OUT/'status.json',dict(stage='features',done=len(rows),total=len(p['cases']),elapsed_seconds=time.perf_counter()-start))
 train=[x for x in rows if x['case']['split']=='calibration' and x['case']['speed']>0 and x['features']['observable']]
 threshold=.5*min(x['features']['maximum_displacement'] for x in train) if train else 0.
 for x in rows:
  f=x['features'];x['decision']='uncertain' if not f['observable'] else ('stationary' if f['maximum_displacement']<threshold else 'moving');x['exact_decision']='stationary' if f['exact'] else 'uncertain'
 groups={}
 for split in ('calibration','evaluation'):
  for texture in ('natural','flat'):
   for noise in (0.,1.):
    for contrast in (1.,.25):
     for moving in (False,True):
      cohort=[x for x in rows if x['case']['split']==split and x['case']['texture']==texture and x['case']['noise_sigma']==noise and x['case']['contrast']==contrast and (x['case']['speed']>0)==moving]
      assert cohort;key=f'{split}_{texture}_n{noise}_c{contrast}_{"moving" if moving else "still"}';groups[key]=dict(count=len(cohort),stationary=sum(x['decision']=='stationary' for x in cohort),moving=sum(x['decision']=='moving' for x in cohort),uncertain=sum(x['decision']=='uncertain' for x in cohort),exact_stationary=sum(x['exact_decision']=='stationary' for x in cohort))
 evaluation=[x for x in rows if x['case']['split']=='evaluation'];moving=[x for x in evaluation if x['case']['speed']>0];false=sum(x['decision']=='stationary' for x in moving)
 checks={'moving_false_stationary_rate_le_5pct':false/len(moving)<=.05}
 for texture in ('natural','flat'):
  for noise in (0.,1.):
   for contrast in (1.,.25):
    motion=[x for x in evaluation if x['case']['texture']==texture and x['case']['noise_sigma']==noise and x['case']['contrast']==contrast and x['case']['speed']>0]
    checks[f'{texture}_noise_{noise}_contrast_{contrast}_false_le_5pct']=sum(x['decision']=='stationary' for x in motion)/len(motion)<=.05
    if texture=='natural':
     still=[x for x in evaluation if x['case']['texture']==texture and x['case']['noise_sigma']==noise and x['case']['contrast']==contrast and x['case']['speed']==0]
     checks[f'natural_noise_{noise}_contrast_{contrast}_recall_ge_50pct']=sum(x['decision']=='stationary' for x in still)/len(still)>=.5
 report=dict(complete=True,threshold=threshold,calibration_observable_moving=len(train),checks=checks,preliminary_gate=all(checks.values()),moving_total=len(moving),moving_false_stationary=false,
  groups=groups,rows=rows,sources=sources,elapsed_seconds=time.perf_counter()-start,protocol_sha256=sha(OUT/'protocol.json'),caveat=p['caveat'])
 save(OUT/'report.json',report);print(json.dumps(dict(threshold=threshold,checks=checks,moving_false_stationary=false,moving_total=len(moving),evaluation_groups={k:v for k,v in groups.items() if k.startswith('evaluation')}),ensure_ascii=False))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['register','run']);args=ap.parse_args();{'register':register,'run':run}[args.stage]()
