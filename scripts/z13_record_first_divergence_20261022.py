"""Analyze saved prefix trace; one-row counterfactual, no new model or HOTA."""
from pathlib import Path
import sys
import subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_center_hota_20261021 import read,write,sha,now
from scripts.z13_first_divergence_20261022 import RUN,OLD,VIDEO,LAST,IDENT,journal
from scripts.z13_hota_deadline_guard import WindowsProcesses


def main():
    import numpy as np
    from ultralytics.trackers.utils import matching
    import cv2
    from PIL import Image,ImageDraw,ImageFont
    if (RUN/'report.json').exists():raise RuntimeError('Existing analysis must be preserved')
    result=read(RUN/'trace_controller_result.json');assert result['worker_ended'] and result['exit_code']==0 and not result['hard_timeout']
    ps=WindowsProcesses();owned=read(RUN/'owned_processes.json');assert not {p:b for p,b in owned.items() if ps.creation(int(p))==b}
    trace=read(RUN/'trace.json');assert trace['prefix_bit_exact_stand'] and trace['prefix_bit_exact_R1'] and trace['frames']==192
    protocol=read(RUN/'protocol.json');assert trace['protocol_sha256']==sha(RUN/'protocol.json')
    for n,h in protocol['sha256'].items():assert sha(ROOT/n)==h,n
    case=trace['traces']['183']['R1'];before=case['before'];after=case['applied'];stand=trace['traces']['183']['stand']['applied']
    i=next(i for i,t in enumerate(after['pool']) if t['id']==6)
    j=next(j for j,t in enumerate(after['detections']) if t['idx']==2)
    assert [t['id'] for t in before['pool']]==[t['id'] for t in after['pool']]
    assert before['detections']==after['detections']
    pre=np.array(before['pool'][i]['mean']);post=np.array(after['pool'][i]['mean'])
    assert np.array_equal(pre[2:],post[2:])
    costs=np.asarray(after['cost'],np.float32);restored=costs.copy();restored[i]=np.asarray(before['cost'][i],np.float32)
    assert np.array_equal(np.delete(costs,i,axis=0),np.delete(restored,i,axis=0))
    m,_,_=matching.linear_assignment(restored,thresh=after['threshold'])
    assert [i,j] in np.asarray(m).tolist() and not any(y==j for x,y in after['matches'])
    assert [i,j] in before['matches']
    response=next(x for x in case['responses'] if x['id']==6)
    assert response['confidence']>=response['tau']
    alignment=read(ROOT/f'runs/train_cache/{VIDEO}/alignment.json')['matches']
    truth=read(ROOT/f'data/bdd100k/labels/box_track_20/train/{VIDEO}.json')
    assert not any(m['video_frame_index']==183 for m in alignment)
    annotations=[]
    for frame in (179,185,191):
        match=next(m for m in alignment if m['video_frame_index']==frame)
        item=next(f for f in truth if f['name']==match['reference_name'])
        annotations.append(dict(raw_frame=frame,reference_name=item['name'],object=next(x for x in item['labels'] if x['id']=='00023512')))
    def iou(a,b):
        lo=np.maximum(a[:2],b[:2]);hi=np.minimum(a[2:],b[2:]);intersection=float(np.prod(np.maximum(hi-lo,0)))
        return intersection/float(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-intersection)
    with np.load(OLD/f'R1/signatures/{VIDEO}/signatures.npz') as z:r1=z['outputs'].copy()
    with np.load(ROOT/f'runs/fly_signatures/{VIDEO}/signatures.npz') as z:base=z['outputs'].copy()
    consequences=[]
    for f in range(183,192):
        target=base[(base[:,0]==f)&(base[:,5]==6)][0];rows=r1[r1[:,0]==f]
        closest=max(rows,key=lambda x:iou(x[1:5],target[1:5]))
        consequences.append(dict(frame=f,stand_id=6,R1_nearest_id=int(closest[5]),iou_to_stand=iou(closest[1:5],target[1:5])))
    gt185=annotations[1]['object']['box2d'];gtbox=np.array([gt185[k] for k in ('x1','y1','x2','y2')])
    row185=r1[(r1[:,0]==185)&(r1[:,5]==155)][0]
    assert iou(row185[1:5],gtbox)>.9
    control_prefix={}
    for arm in ('C','D'):
        with np.load(OLD/arm/'signatures'/VIDEO/'signatures.npz') as z:out=z['outputs']
        control_prefix[arm]=bool(np.array_equal(out[out[:,0]<=LAST],base[base[:,0]<=LAST]))
    delta=post[:2]-pre[:2]
    report=dict(verified=now().isoformat(),video=VIDEO,raw_frame=183,prefix_frames=192,prefix_stand_bit_exact=True,prefix_R1_bit_exact=True,
        all_owned_processes_ended=True,source_sha256={n:sha(ROOT/n) for n in ['runs/z13_first_divergence_20261022/trace.json','runs/z13_first_divergence_20261022/protocol.json','tracking/z13_ridge_localizer.py']},
        track_id=6,center_before=pre[:2].tolist(),center_after=post[:2].tolist(),shift=delta.tolist(),shift_pixels=float(np.linalg.norm(delta)),
        confidence=response['confidence'],tau=response['tau'],eye_center=response['eye_center'],eye_size=391,
        correct_candidate=dict(raw_detection_index=2,box=after['detections'][j]['box'],score=after['detections'][j]['score'],cost_before=before['cost'][i][j],cost_after=after['cost'][i][j],threshold=after['threshold']),
        counterfactual=dict(only_track6_cost_row_restored=True,all_other_rows_unchanged=True,track6_matches_detection2=True),
        consequences=consequences,nearby_real_annotations=annotations,gt_at183=False,gt_interpolated=False,
        R1_id155_iou_to_GT185=iou(row185[1:5],gtbox),other_saved_prefix_equal_to_stand=control_prefix,
        conclusion='One reproduced R1 center shift blocks recovery of track6 at183; restoring only its association-cost row recovers the same candidate. At184 the object is output with newID155, confirmed against real GT185. Not a switch to a neighboring identity at183, but a missed recovery and fragmentation.',
        limits='Post-hoc worst R1 video; first divergence only, not the entire HOTA loss or C/D. No GT183. Counterfactual isolates this association, not full later trajectories. No claim that every boundary peak is wrong, no proposed gate accepted, no new HOTA.')
    write(RUN/'report.json',report)
    capture=cv2.VideoCapture(str(ROOT/f'data/bdd100k/videos/train/{VIDEO}.mov'));capture.set(cv2.CAP_PROP_POS_FRAMES,183);ok,frame=capture.read();capture.release();assert ok
    img=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).crop((420,225,660,395)).resize((1200,850))
    draw=ImageDraw.Draw(img);font=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',23)
    def point(p):return ((float(p[0])-420)*5,(float(p[1])-225)*5)
    for box,col in [(before['pool'][i]['box'],'#13AA57'),(after['pool'][i]['box'],'#EF3038'),(after['detections'][j]['box'],'#2078FF')]:
        draw.rectangle((*point(box[:2]),*point(box[2:])),outline=col,width=5)
    p1=point(pre[:2]);p2=point(post[:2]);draw.line((p1,p2),fill='#EF3038',width=7);draw.ellipse((p2[0]-7,p2[1]-7,p2[0]+7,p2[1]+7),fill='#EF3038')
    draw.rectangle((0,0,1200,100),fill='white');draw.text((15,6),'Кадр183: зелёный — прогноз, синий — детекция, красный — поправка R1',font=font,fill='black')
    draw.text((15,43),'Стоимость сопоставления: 0,123 → 1,000 при допустимом максимуме0,8',font=font,fill='black')
    img.save(RUN/'case183.png')
    cat=read(ROOT/'experiments/project_history.json');card=next(c for c in cat['experiments'] if c['id']==IDENT)
    card.update(state='completed_positive',done='2026-10-22',summary='Диагностика воспроизведена: поправка R1 срывает восстановление трека6; это не положительный результат HOTA.',
        obtained='runs/z13_first_divergence_20261022/report.json: оба префикса0–191 побитово совпали. На183 центр6: [569,7998;310,6098]→[491,1008;298,1021], сдвиг79,6867px; балл5,2631≥порог5,1134. correct_candidate: стоимость0,12349→1,0 при допустимых0,8. counterfactual: восстановление только строки6 возвращает сопоставление с детекцией2. consequences: с184 объект получает ID155; на185 IoU с реальной рамкой00023512 больше0,9. Точные значения и реальные соседние аннотации — в первичном JSON.',
        interpretation='Причина первого расхождения подтверждена для этого случая: уверенный пик почти на левой границе вырезки отодвинул хороший прогноз, сорвал восстановление и фрагментировал траекторию. Перехвата соседнего объекта на183 нет; возник пропуск восстановления. Причина суммарных потерь35видео этим не установлена.',
        decision='Следующее действие должно исправлять допуск ошибочной поправки, сохраняя центрированный считыватель. Сначала проверить на train, насколько повторяется граничный/несогласованный пик, и только после заранее заданного контроля выбирать замену оценки отказа. По этому dev-примеру порог не настраивать.')
    for a in card['acceptance']:a.update(met=True,evidence='runs/z13_first_divergence_20261022/report.json')
    card['evidence'] += [dict(kind='file',ref='runs/z13_first_divergence_20261022/'+n,claim='Побитовое воспроизведение и наблюдение первого события') for n in ['trace.json','report.json','trace_controller_result.json','case183.png']]
    card['commits']=[subprocess.check_output(['git','--no-optional-locks','rev-parse','HEAD'],cwd=ROOT,text=True).strip()]
    write(ROOT/'experiments/project_history.json',cat)
    journal('разобрано первое расходящееся сопоставление', '*Получено.* '+card['obtained']+'\n\n*Вывод.* '+card['interpretation']+'\n\n*Решение.* '+card['decision']+'\n\nЭто положительный исход диагностической проверки, не улучшение алгоритма. НоваяHOTA не измерялась, процент прироста и доля потолка неприменимы. Существующие полные результаты неизменны. Формулировки проверены на предмет/инструмент, повтор и соответствие реально выполненному.')
    print('One-row counterfactual recovers track6. Fragmentation6→155 confirmed; no GT at183.')
    print('Shift',report['shift_pixels'],'GT185 IoU',report['R1_id155_iou_to_GT185'],'C/D prefix',control_prefix)


if __name__=='__main__':main()
