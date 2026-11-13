import cv2,datetime,hashlib,json,numpy as np
from pathlib import Path
ROOT=Path(r'E:\Dissertation');OUT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,d):(OUT/n).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
cases=[
 dict(id='B_day_shop',video='b20b69d2-ca16c907',event_frame=543,reference_frame=537,query_frame=543,target_ref_raw_id=21381,competitor_ref_raw_id=21388,query_raw_id=21638,competitor_query_raw_id=21653,
      target_description='Бордовая машина перед магазином, слева от большого чёрного пикапа; в запросе часть закрыта ближним столбом.',competitor_description='Белый компактный автомобиль перед тем же магазином, слева от бордовой машины.',target_gt_id='00128254',competitor_gt_id='00128251',detail_bounds=[870,300,1080,375],anchors=[532,538,544],
      scope='Новая диагностическая пара на кадрах резервного события543; целевая ROI сохранена из исходного предложения, второй физический автомобиль выбран по видимости до исходов. Не вывод об исправлении старого отказа.'),
 dict(id='B_night_rain',video='b23f7012-fab06dac',event_frame=534,reference_frame=529,query_frame=534,target_ref_raw_id=10515,competitor_ref_raw_id=10510,query_raw_id=10759,competitor_query_raw_id=10757,
      target_description='Тёмный автомобиль справа с белой номерной площадкой, часть левой стороны закрыта соседней машиной; уверенная часть — видимая задняя панель.',competitor_description='Синий/тёмный автомобиль слева с жёлтой номерной площадкой и двумя яркими задними фонарями.',target_gt_id='00117875',competitor_gt_id='00117874',detail_bounds=[670,325,1080,455],anchors=[529,535],
      scope='Новая диагностическая пара на кадрах резервного события534; исходные детекторные ROI, физическая проверка по последовательности. Не оценка продолженного трекера.'),
 dict(id='B_dusk_queue',video='b216243d-ad4306b9',event_frame=150,reference_frame=145,query_frame=150,target_ref_raw_id=11594,competitor_ref_raw_id=11592,query_raw_id=11998,competitor_query_raw_id=11997,
      target_description='Ближняя чёрная машина справа от светлого автомобиля, с жёлтой номерной площадкой; уверенная часть ниже заднего стекла.',competitor_description='Крупный светлый автомобиль слева на переднем плане, с широкой светлой задней панелью.',target_gt_id='00121867',competitor_gt_id='00121859',detail_bounds=[510,225,840,420],anchors=[145,151],
      scope='Новая диагностическая пара крупных видимых машин на кадрах резервного события150. Это не исходная крошечная цель old_id215 и не проверка исправления её сохранённого отказа.'),
]
before=dict(frozen_at=datetime.datetime.now().astimezone().isoformat(),outcomes_read=False,cases=cases,
 revision_reason='После просмотра исходных кадров уточнена единица выборки: три диагностические физические пары на кадрах резерва B. Не требуется воспроизведение спорной личности исходного old_id. Основной агент согласовал этот предел до новых оценок. Не все 25 предложения пригодны для ручного разделения; остальные не объявляются успехами или неудачами.',
 prior_protocol='selection_protocol.json',prior_protocol_sha256=sha(OUT/'selection_protocol.json'),
 unchanged_exclusion='Все три видео взяты из заранее отделённого B; прежние5подборочных видео и2179не входят. Видео ранее участвовали в агрегированной проверке проекта, поэтому это новые для выбора масок случаи, не глобально невидимая финальная выборка.',
 negative_visibility_witness=dict(video='b23adb0d-72704b27',event_frame=606,reference_frame=600,panel='candidate_panels/b23adb0d-72704b27_606.png',status='no_observable_target_in_original_query_roi',reason='На исходном запросе ROI у левого края не содержит различимой видимой части прежней машины; сравнение сходства не проводится. Это ограничение наблюдаемости, не успех масок.'),
 acceptance='Никаких новых случаев или изменений масок после YOLO26-seg/MASA-оценок. Неизвестная видимость и пропущенные/неоднозначные автоматические маски остаются в знаменателе попыток. Ручные маски — консервативные части, без заявления полной разметки. Полные и маскированные варианты используют одинаковые исходные ROI.')
write('frozen_pairs.json',before)
source=json.loads((ROOT/'research_identity_holdout_20261112/review/selected_events_B_RESERVED_v2.json').read_text(encoding='utf8'))
disposition=[]
for e in source['events']:
 selected=next((c for c in cases if c['video']==e['video'] and c['event_frame']==e['frame']),None)
 status='selected_diagnostic_frame_pair' if selected else ('visibility_witness_only' if e['video']=='b23adb0d-72704b27' and e['frame']==606 else 'not_selected_no_outcome_evaluation')
 reason='One fixed physical pair in this video; other reserved proposals retained without outcome-based selection or claims.'
 if e['video']=='b23adb0d-72704b27':reason='No reliable fourth physical pair frozen in this bounded visual diagnostic; dark/small/truncated proposals retained, not scored.'
 disposition.append(dict(event_key=e['event_key'],status=status,reason=reason))
write('selection_disposition_25.json',dict(records=disposition,total=25,selected=3,visibility_witness=1,other_not_evaluated=21,outcomes_read=False))
(OUT/'continuity').mkdir(exist_ok=True);(OUT/'anchors').mkdir(exist_ok=True)
alignment=[]
for c in cases:
 v=c['video'];cap=cv2.VideoCapture(str(ROOT/f'data/bdd100k/videos/val/{v}.mov'));imgs={}
 clock=json.loads((ROOT/f'runs/identity_assignment_20261108/full_precision/runtime_inputs/{v}.json').read_text(encoding='utf8'))['anchors']
 gt={x['name']:x for x in json.loads((ROOT/f'data/bdd100k/labels/box_track_20/val/{v}.json').read_text(encoding='utf8'))}
 needed=set(range(c['reference_frame'],c['query_frame']+2))
 for anchor in c['anchors']:needed.update([anchor-1,anchor,anchor+1])
 for f in sorted(needed):
  cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,img=cap.read();assert ok;imgs[f]=img
  if c['reference_frame']<=f<=c['query_frame']+1:cv2.imwrite(str(OUT/f'continuity/{c["id"]}_{f}.png'),img)
 cap.release();tiles=[]
 x1,y1,x2,y2=c['detail_bounds']
 for f in range(c['reference_frame'],c['query_frame']+2):
  crop=imgs[f][y1:y2,x1:x2];crop=cv2.resize(crop,(640,int(round(crop.shape[0]*640/crop.shape[1]))),interpolation=cv2.INTER_NEAREST)
  tile=np.full((420,660,3),25,np.uint8);tile[30:30+crop.shape[0],10:650]=crop
  cv2.putText(tile,f'{c["id"]} raw {f}',(10,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1);tiles.append(tile)
 while len(tiles)%2:tiles.append(np.zeros_like(tiles[0]))
 panel=np.concatenate([np.concatenate(tiles[i:i+2],axis=1) for i in range(0,len(tiles),2)],axis=0)
 cv2.imwrite(str(OUT/f'continuity/{c["id"]}_contact.png'),panel)
 for anchor in c['anchors']:
  item=clock[str(anchor)];original=ROOT/f'data/bdd100k/images/track/val/{v}'/item['name'];jpg=cv2.imread(str(original));assert jpg is not None
  errors=[dict(raw_frame=f,mean_absolute_pixel_error=float(np.abs(jpg[y1:y2,x1:x2].astype(float)-imgs[f][y1:y2,x1:x2]).mean())) for f in [anchor-1,anchor,anchor+1]]
  labels=[o for o in gt[item['name']]['labels'] if str(o['id']) in [c['target_gt_id'],c['competitor_gt_id']]]
  alignment.append(dict(case_id=c['id'],anchor=anchor,original_jpg=str(original.relative_to(ROOT)).replace('\\','/'),jpg_sha256=sha(original),errors=errors,best_raw_frame=min(errors,key=lambda x:x['mean_absolute_pixel_error'])['raw_frame'],labels=labels))
  overlay=jpg.copy()
  for o in labels:
   b=o['box2d'];box=tuple(int(b[k]) for k in ['x1','y1','x2','y2']);color=(0,255,0) if o['id']==c['target_gt_id'] else (255,100,0)
   cv2.rectangle(overlay,box[:2],box[2:],color,1);cv2.putText(overlay,o['id'],(box[0],box[1]-3),cv2.FONT_HERSHEY_SIMPLEX,.35,color,1)
  cv2.imwrite(str(OUT/f'anchors/{c["id"]}_{anchor}_gt.png'),overlay)
write('identity_anchor_evidence.json',dict(records=alignment,physical_identity_method='Raw image sequence and distinctive visible features; GT continuity only corroboration, never IoU ownership of visible pixels.',all_clocks_best_at_expected=all(x['best_raw_frame']==x['anchor'] for x in alignment)))
print(json.dumps({'frozen':3,'anchor_checks':len(alignment),'all_clocks_best_at_expected':all(x['best_raw_frame']==x['anchor'] for x in alignment)}))
