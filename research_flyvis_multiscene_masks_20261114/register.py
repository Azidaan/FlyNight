from common import *
import gzip,subprocess,shutil
BASE=ROOT/'runs/domain_matched_native_20261109'
known={'b1ca2e5d-84cf9134','b20b69d2-ca16c907','b216243d-ad4306b9','b23f7012-fab06dac'}
videos=sorted(p.name for p in (BASE/'features').iterdir() if p.is_dir())
test=sorted(v for v in videos if v not in known)[::4][:4]
records=[];sources={}
for video in videos:
 maps={}
 for path in sorted((BASE/'features'/video).glob('*.C34.json')):
  m=read(path);key=(m['sequence'][-1][0],tuple(m['sequence'][-1][1]));maps.setdefault(key,[]).append((path,m))
 log=BASE/'rollouts/C34'/f'{video}.events.json.gz';sources[rel(log)]=sha(log)
 events=json.load(gzip.open(log,'rt',encoding='utf8'))['events'];candidates={}
 for e in events:
  for p in e['proposals']:
   for frame,crop,box,role in [(p['reference_frame'],p['reference_crop'],p['reference_box'],'reference'),(e['frame'],p['current_crop'],p['current_box'],'query')]:
    key=(frame,tuple(crop))
    if key not in maps:continue
    path,m=maps[key][0]
    candidates.setdefault(path.stem,dict(video=video,frame=frame,crop=crop,box=box,role=role,metadata=rel(path),map=rel(path.with_suffix('.npy')),sequence=m['sequence']))
 rows=sorted(candidates.values(),key=lambda r:(r['frame'],r['metadata']))
 unique={}
 for r in rows:unique.setdefault(r['frame'],r)
 rows=list(unique.values())
 if video in test:
  # Select visible-sized windows using geometry alone, never teacher/reader masks.
  ranked=sorted(rows,key=lambda r:(-(r['crop'][2]-r['crop'][0]),r['frame']))
  selected=[]
  for r in ranked:
   if all(abs(r['frame']-s['frame'])>=30 for s in selected):selected.append(r)
   if len(selected)==2:break
  assert len(selected)==2,(video,len(selected))
 else:
  count=min(8,len(rows));indices=sorted(set(round(i*(len(rows)-1)/max(1,count-1)) for i in range(count)));selected=[rows[i] for i in indices]
 for r in selected:
  r['split']='test' if video in test else 'train';r['id']=f"{video}_{r['frame']}";r['image']=rel(OUT/'frames'/f"{r['id']}.png")
  m=read(ROOT/r['metadata']);assert sha(ROOT/r['map'])==m['sha256'];r['retinal']=[]
  for f,c in r['sequence']:
   key=hashlib.sha256(json.dumps([int(f),list(c)],separators=(',',':')).encode()).hexdigest()[:32]
   options=[n for n in m['retinal_sources'] if Path(n).stem==key];assert options,(r['id'],key)
   n=options[0];assert sha(ROOT/n)==m['retinal_sources'][n];sources[n]=sha(ROOT/n);r['retinal'].append(dict(frame=f,crop=c,path=n))
  for n in [r['metadata'],r['map']]+([m['reused_from']] if 'reused_from' in m else []):sources[n]=sha(ROOT/n)
  records.append(r)
 del events
legacy=read(ROOT/'runs/identity_assignment_20261108/full_precision/protocol.json')
for v in videos:
 n=legacy['videos'][v]['video_path'];sources[n]=sha(ROOT/n);assert sources[n]==legacy['videos'][v]['sources'][n]
for n in ['AGENTS.md','runs/z13_local_peak_20261021/geometry.json','weights/yolo26seg_official_20261114/yolo26n-seg.pt','runs/flyvis/results/flow/0000/000/best_chkpt','tracking/retina.py',rel(Path(__file__)),rel(OUT/'common.py')]:sources[n]=sha(ROOT/n)
criteria=['Training/test videos disjoint, frozen before teacher predictions. Test clips have no prior manual mask reader fit; historical unrelated tracking use disclosed.','Same retinal sequence, ROI, coordinate features, normalization fit only on training, reader and lambda=1 for FlyVis and no-FlyVis control. No core/fulltracker changes.','Manual conservative test cores and definite non-target regions drawn on original images before reader predictions; unknown ignored. Automatic teacher labels remain pseudo-labels, not ground truth.','Independent recomputation of saved predictions/metrics and no source changes.','Mask feasibility: mean per-video known-region target IoU>=0.5, all four test videos>=0.3. Additional FlyVis benefit: positive mean difference vs matched retinal history and positive difference in at least three of four videos.','Report teacher misses/merges and all heldout cases; no test-directed hyperparameter or threshold selection.']
protocol=dict(registered=now(),head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),records=records,train_videos=[v for v in videos if v not in test],test_videos=test,inputs=sources,
 question='Does a frozen target-conditioned mask reader trained across scenes transfer to videos excluded from its training, compared with the same reader without FlyVis?',
 labels='YOLO26n-seg pseudo-labels matched to the existing detector box by maximum bbox IoU, require IoU>=0.5 and >=80% of selected mask inside supplied ROI. Training masks may contain teacher errors; manual QA flags excluded before fit. Evaluation uses independent manually reviewed visible cores; incomplete contours disclosed.',
 task='Binary visible target vs other, conditioned on existing same-frame detector ROI. This changes old scene-specific red/black labels to transferable target/background classes. It is a conditional segmentation diagnostic, not identity recovery or full tracking.',
 reader=dict(kind='Same class-balanced ridge lambda1 with unpenalized intercept as prior probe; binary classes',regularization=1.0,normalization='training accepted cells only; per-arm mean/SD; SD<1e-12=>1',decision='argmax two scores',geometry='6 shared features: normalized relative x,y,x^2,y^2,x*y,inside detector ROI',training_cell='positive mask footprint>=0.9, negative<=0.1; intermediate ignored',weighting='equal video, equal frame within video, equal class within frame'),
 arms=['FlyVis34_G6','Retina_history34_G6','Geometry6','RGB3_G6','FlyVis34','Retina_history34'],
 learning_sizes=['one earliest accepted frame per training video','all accepted training frames'],
 teacher=dict(imgsz=960,conf=0.25,iou=0.7,max_det=300,retina_masks=True,rect=True,half=False,batch=1,classes=[2],device=0,verbose=False,save=False),
 acceptance=criteria,scope='Cached flow-pretrained frozen FlyVis000; no new core training. New test means unseen by this mask reader, not unused by every historical project experiment. Same nonlinear Lamina retina/history on both arms. Conservative known-region IoU is not full silhouette IoU.',
 stop='Stop for failed provenance/alignment, nonfinite values, any train/test video overlap, fewer than8 accepted training videos, or <3 positive and3 negative cells per training example. Excluded examples retained in report.',
 budget='Single small teacher GPU pass <=20minutes; CPU reader and audit <=20minutes; artifacts expected under250MiB, all E. No new downloads or environment changes.',deadline='2026-11-16T07:35:00+03:00',counterargument='The geometry or retinal history alone may explain mask quality; noisy teacher masks can cap performance. This test does not isolate data volume if compared with old scene-specific three-class task.')
save('protocol.json',protocol)
(OUT/'before').mkdir(exist_ok=False)
for n in ['project_history.json','system_problems.json','obsidian_experiment_topics.json','useful_literature.json']:shutil.copyfile(ROOT/'experiments'/n,OUT/'before'/n)
shutil.copyfile(ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md',OUT/'before/ЕДИНЫЙ_ПЛАН.md')
cat=read(ROOT/'experiments/project_history.json');identity='flyvis_multiscene_masks_20261114';assert not any(x['id']==identity for x in cat['experiments'])
cat['experiments'].append(dict(id=identity,title='Маска заданной машины — обучение считывателя FlyVis на разных сценах',area='own_tracker',state='running',started='2026-11-14',question=protocol['question'],hypothesis='Большее число сцен может улучшить перенос считывателя маски; равный контроль отделяет вклад карт FlyVis от входа и геометрии.',expected='Обученные веса, независимые видео, маски и сравнение с тем же считывателем без FlyVis.',acceptance=[dict(criterion=c,met=None) for c in criteria],evidence=[dict(kind='file',ref=rel(OUT/'protocol.json'))]))
(ROOT/'experiments/project_history.json').write_text(json.dumps(cat,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
t=read(ROOT/'experiments/obsidian_experiment_topics.json');t['topics'][identity]=['Сохранение личности','Содержание области наблюдения','Обучение маскам по разным сценам'];(ROOT/'experiments/obsidian_experiment_topics.json').write_text(json.dumps(t,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
journal('Постановка (`'+identity+'`). Учебные и проверочные видео разделены до предсказаний учителя и считывателя; протокол '+rel(OUT/'protocol.json')+'. Обучается условная маска заданной рамкой машины: ядро заморожено, контроль получает ту же историю входа и геометрию. Автоматические маски учителя обозначены как шумная разметка; ручная проверка новых видео отделена от обучения. Полный трекер не запускается.')
print(json.dumps(dict(train_frames=sum(r['split']=='train' for r in records),test_frames=sum(r['split']=='test' for r in records),train_videos=len(videos)-len(test),test_videos=test),ensure_ascii=False))
