"""A bounded, saved-training-data gate for detection-indexed FlyVis readout."""
from pathlib import Path
from datetime import timedelta
import json
import os
import subprocess
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
RUN=ROOT/'runs/z13_detection_candidate_gate_20261024'
OLD=ROOT/'runs/z13_candidate_reserve_20261021'
W=ROOT/'runs/z13_weekend_20261024'
IDENT='z13_detection_candidate_gate_v1'


def candidates(boxes,scores,pred,crop):
    """No labels: nearest three high-score detection centers inside current crop."""
    boxes=np.asarray(boxes,dtype=np.float64).reshape(-1,4)
    scores=np.asarray(scores,dtype=np.float64);pred=np.asarray(pred,dtype=np.float64)
    assert scores.shape==(len(boxes),) and pred.shape==(2,)
    assert np.isfinite(boxes).all() and np.isfinite(scores).all() and np.isfinite(pred).all()
    centers=(boxes[:,:2]+boxes[:,2:])/2
    x1,y1,x2,y2=map(float,crop)
    keep=np.flatnonzero((scores>=.25)&(boxes[:,2]>boxes[:,0])&(boxes[:,3]>boxes[:,1])&
        (centers[:,0]>=x1)&(centers[:,0]<=x2)&(centers[:,1]>=y1)&(centers[:,1]<=y2))
    order=np.argsort(np.linalg.norm(centers[keep]-pred,axis=1),kind='stable')
    ids=keep[order]
    return ids[:3],centers[ids[:3]],centers[ids]


def journal(title,text):
    from scripts import z13_val20_common_20261024 as S
    saved=S.IDENT
    try:S.IDENT=IDENT;S.journal(title,text)
    finally:S.IDENT=saved


def prepare():
    assert not RUN.exists()
    old=read(OLD/'protocol.json');videos=old['evaluate']
    names=[Path(__file__).relative_to(ROOT).as_posix(),'tests/test_z13_detection_candidate_gate.py',
        'scripts/z13_detection_candidate_gate_console.ps1','scripts/z13_bounded_stage_20261022.py',
        'scripts/z13_continue.py','runs/z13_candidate_reserve_20261021/protocol.json',
        'runs/z13_candidate_reserve_20261021/report.json','runs/z13_candidate_reserve_20261021/evaluate/labels.json']
    for v in videos:names.extend([f'runs/z13_data/{v}.npz',f'runs/train_cache/{v}/detections/raw_video.npz'])
    for arm in ('D','C','R1'):
        names += [x.relative_to(ROOT).as_posix() for x in (OLD/'evaluate'/arm).glob('cache_*.npz')]
    auth=read(W/'authorization.json')
    p=dict(id=IDENT,created=now().isoformat(),videos=videos,device='cpu',budget_seconds=600,
        save_at=min(now()+timedelta(seconds=600),__import__('datetime').datetime.fromisoformat(auth['save_at'])).isoformat(),
        hard_stop=min(now()+timedelta(seconds=620),__import__('datetime').datetime.fromisoformat(auth['hard_stop'])).isoformat(),
        action='Replace the output candidates of a future FlyVis readout:three actual detection centers,not three arbitrary response peaks. First measure saved-data conditional localization reserve;no model is trained in this gate.',
        candidate_rule='Raw unchanged YOLO26 boxes confidence>=0.25 (existing first-association threshold),positive area,center inside same saved square crop;nearest3 to current saved Kalman prediction,stable original order for ties. No class filtering or GT in generation. At most3 candidates,as in old peak readout. No cap variants or threshold search.',
        primary='All26 repeated-train videos and all8326 saved labels align exactly. Detection3 oracle useful fraction minus oldD peak3 oracle:mean>=0.02 and paired95%lower>0. If upper<0.02 stop this candidate set;otherwise inconclusive. This is only an oracle availability gate,not an attainable gain or HOTA.',
        useful='Existing kind1 rows only:Euclidean center error<=0.25*max(true box width,height) AND strictly less than unchanged saved prediction error. Empty set is nonuseful. GT only computes outcomes. Unknown and absent rows get no fabricated identity.',
        controls='Same3actualdetections for any later C/D/R1 and geometry-only learned readout;nearest-detection policy withoutFlyVis measured now,all-detection oracle secondary only. No downstream fitting until separate protocol and extraction price gate.',
        difference='All previous readouts chose up to3maxima of the fitted retinal response. This changes the candidate support to external unchanged detector proposals;it does not claim the detector or tracker improved. Future learning must compare activity versus same observed geometry/confidence inputs.',
        counterargument='Correct detection may already be selected by geometry;all gain may come from YOLO proposals,notFlyVis. Oracle ignores multi-track competition;class-unfiltered square-crop support is not identical to the hexagonal feature field. Projection/descriptor extraction and missing-candidate refusal remain untested. Sparse labels,perturbed training crops,and repeatedly used26train do not establish closed-loop transfer.',
        paper=dict(url='https://arxiv.org/html/2105.11595v1',sections='2.2,3.1,3.4,5.1',
            scope='SiamMOT supports matching the motion/region output to association;it does NOT implement this proposed detection3 decoder. New candidate support is our hypothesis,not a reproduction or published performance claim.'),
        sha256={n:sha(ROOT/n) for n in names})
    write(RUN/'protocol.json',p)
    write(RUN/'status.json',dict(state='ready',stage='Доступность трёх рамок детектора',updated=now().isoformat(),done=0,total=26))
    cp=ROOT/'experiments/project_history.json';cat=read(cp);assert not any(c['id']==IDENT for c in cat['experiments'])
    cat['experiments'].append(dict(id=IDENT,title='З13: запас локализации при чтении трёх детекций вместо пиков',area='flyvis',state='ready',started=now().strftime('%Y-%m-%d'),tags=['fly-problem-P8'],
        question='Есть ли дополнительный запас полезных центров для смены пространства кандидатов считывателя?',
        hypothesis='Реальные рамки могут содержать полезный центр,отсутствующий среди3пиков;самой мухе этот запас может быть недоступен.',
        expected=p['action']+' '+p['candidate_rule'],acceptance=[dict(criterion=p['primary'],met=None)],
        summary='Диагностическая CPU-проверка до600с зарегистрирована до расчёта.',obtained='',interpretation='',decision='',cannot_claim=p['counterargument'],
        metrics=[],commits=[],files=[names[0]],evidence=[dict(kind='file',ref=(RUN/'protocol.json').relative_to(ROOT).as_posix(),claim='Кандидаты,критерий и ограничения до исхода')]))
    write(cp,cat)
    journal('постановка смены кандидатов:три детекции вместо трёх пиков',
        '*Постановка.* '+p['action']+'\n\n*Критерий до расчёта.* '+p['primary']+'\n\n'
        '*Контроли и возражение.* '+p['controls']+' '+p['counterargument']+'\n\n'
        'Изменяется потенциальный узел считывания,не YOLO/ByteTrack/MASA. SiamMOT§2.2/3.1/3.4/5.1,'
        'https://arxiv.org/html/2105.11595v1:подтверждает роль регионального движения в ассоциации,'
        'но не нашу конкретную схему3детекций. Только сохранённые train26,безGPU/видео/новойHOTA/обучения/75. '
        'Бюджет600с,независимый контроллер и отдельная видимая консоль. Ни одно значение порога не выбирается поval20.')


def worker():
    from scripts.z13_localization_labels_20261023 import verify_sources
    from scripts.z13_continue import paired
    started=time.monotonic();p=read(RUN/'protocol.json');verify_sources(p)
    labels=read(OLD/'evaluate/labels.json');assert len(labels)==8326
    lookup={}
    for arm in ('D','C','R1'):
        cache={}
        for path in sorted((OLD/'evaluate'/arm).glob('cache_*.npz')):
            with np.load(path) as z:
                for i in range(len(z['t'])):cache[(int(z['episode'][i]),int(z['t'][i]))]=z['centers'][i,:int(z['count'][i])].copy()
        lookup[arm]=cache
    byvideo={v:[l for l in labels if l['video']==v] for v in p['videos']}
    per={k:{} for k in ('detection3_oracle','all_detection_oracle','nearest_detection', 'D_peak3_oracle','C_peak3_oracle','R1_peak3_oracle','D_peak1')}
    rows=[];offset=0
    for vi,v in enumerate(p['videos']):
        assert time.monotonic()-started<600 and not (RUN/'stop.request').exists()
        with np.load(ROOT/f'runs/z13_data/{v}.npz') as z:
            data={k:z[k] for k in ('ep_offset','fr_frame','lab_ep','lab_pos','lab_kind','lab_center','lab_pred','lab_crop')}
        with np.load(ROOT/f'runs/train_cache/{v}/detections/raw_video.npz') as z:
            det={k:z[k] for k in ('xyxy','conf','frame_idx')}
        local={(int(e),int(pos)):i for i,(e,pos) in enumerate(zip(data['lab_ep'],data['lab_pos']))}
        values={k:[] for k in per}
        for l in byvideo[v]:
            e=l['episode']-offset;i=local[(e,l['pos'])]
            assert int(data['lab_kind'][i])==l['kind'] and np.array_equal(data['lab_pred'][i],l['pred'])
            frame=int(data['fr_frame'][int(data['ep_offset'][e])+4+l['pos']])
            ix=np.flatnonzero(det['frame_idx']==frame)
            ids,centers,allcenters=candidates(det['xyxy'][ix],det['conf'][ix],l['pred'],data['lab_crop'][i])
            r=dict(label=l['label'],video=v,frame=frame,kind=l['kind'],candidate_indices=ix[ids].tolist(),centers=centers.tolist(),all_count=len(allcenters))
            if l['kind']==1:
                assert np.array_equal(data['lab_center'][i],l['gt'])
                gt=np.asarray(l['gt']);pred_error=float(np.linalg.norm(np.asarray(l['pred'])-gt))
                def useful(cs):
                    error=np.linalg.norm(np.asarray(cs).reshape(-1,2)-gt,axis=1)
                    return (error<=.25*l['scale'])&(error<pred_error)
                ans=dict(detection3_oracle=bool(useful(centers).any()),all_detection_oracle=bool(useful(allcenters).any()),nearest_detection=bool(useful(centers[:1]).any()))
                for arm in ('D','C','R1'):
                    cs=lookup[arm][(l['episode'],l['pos']+4)];ans[arm+'_peak3_oracle']=bool(useful(cs).any())
                    if arm=='D':ans['D_peak1']=bool(useful(cs[:1]).any())
                for k,x in ans.items():values[k].append(x)
                r['outcomes']=ans
            rows.append(r)
        assert values['D_peak1']
        for k,x in values.items():per[k][v]=float(np.mean(x))
        offset+=len(data['ep_offset'])-1
        write(RUN/'status.json',dict(state='running',stage='Сверка сохранённых обучающих эпизодов',updated=now().isoformat(),done=vi+1,total=26,elapsed_seconds=round(time.monotonic()-started,2),variant=v))
    old=read(OLD/'report.json')
    for arm in ('D','C','R1'):
        assert abs(float(np.mean(list(per[arm+'_peak3_oracle'].values())))-old['arms'][arm]['oracle3_mean'])<1e-12
    assert abs(float(np.mean(list(per['D_peak1'].values())))-old['arms']['D']['top1_mean'])<1e-12
    contrast={k:paired(per[k],per['D_peak3_oracle']) for k in ('detection3_oracle','all_detection_oracle','nearest_detection')}
    main=contrast['detection3_oracle'];passed=main['n']==26 and main['mean_per_video']>=.02 and main['ci95'][0]>0
    state='reserve_sufficient' if passed else 'reserve_below_minimum' if main['ci95'][1]<.02 else 'inconclusive'
    verify_sources(p);write(RUN/'rows.json',rows)
    write(RUN/'report.json',dict(videos=26,rows=len(rows),state=state,primary_pass=passed,paired_vs_D_peak3_oracle=contrast,per_video=per,
        mean={k:float(np.mean(list(v.values()))) for k,v in per.items()},
        old_three_peak_oracles_exactly_reproduced=True,rows_sha256=sha(RUN/'rows.json'),not_hota=True,
        limits=p['counterargument'],duration_seconds=time.monotonic()-started))
    write(RUN/'status.json',dict(state='completed',stage='Запас кандидатов сохранён;это не HOTA',updated=now().isoformat(),done=26,total=26))


def record():
    from scripts.z13_hota_deadline_guard import WindowsProcesses
    from scripts.z13_localization_labels_20261023 import verify_sources
    p=read(RUN/'protocol.json');verify_sources(p);e=read(RUN/'controller_result.json')
    assert e['exit_code']==0 and e['all_owned_ended'] and e['worker_ended'] and not e['remaining']
    ps=WindowsProcesses();assert all(ps.creation(int(pid))!=b for pid,b in e['owned'].items())
    r=read(RUN/'report.json');assert r['rows']==8326 and sha(RUN/'rows.json')==r['rows_sha256']
    cp=ROOT/'experiments/project_history.json';cat=read(cp);c=next(x for x in cat['experiments'] if x['id']==IDENT)
    c.update(state='completed_positive' if r['primary_pass'] else 'completed_negative' if r['state']=='reserve_below_minimum' else 'inconclusive',done=now().strftime('%Y-%m-%d'),
        obtained=(RUN/'report.json').relative_to(ROOT).as_posix()+' → mean/paired_vs_D_peak3_oracle: '+json.dumps(dict(mean=r['mean'],paired=r['paired_vs_D_peak3_oracle']),ensure_ascii=False),
        interpretation='Измерен только условный оракульный запас предложения детекций при известных центрах. Это не работа обученного узла,не вклад FlyVis и не польза стенду.',
        decision='Подготовить только отдельный замер стоимости признаков и равного C/D/R1/геометрия-считывателя.' if r['primary_pass'] else 'Не извлекать признаки этой версии;сохранить полный отрицательный/неубедительный исход и выбрать другой механизм.',
        summary='Проверка доступности кандидатов завершена;'+r['state']+';реализация нового считывания ещё не принята.')
    c['acceptance'][0].update(met=r['primary_pass'],evidence=(RUN/'report.json').relative_to(ROOT).as_posix())
    write(cp,cat)
    journal('итог доступности детекций вместо пиков','*Получено.* '+c['obtained']+'\n\n*Вывод.* '+c['interpretation']+'\n\n*Решение.* '+c['decision']+'\n\nК HOTA пересчёт невозможен;доля потолка стенда неизвестна. Оракульный запас не объявляется практическим улучшением.')
    write(RUN/'finish_recorded.json',dict(at=now().isoformat(),report_sha256=sha(RUN/'report.json'),all_owned_ended=True))
    print(json.dumps(r['paired_vs_D_peak3_oracle'],ensure_ascii=False),flush=True)


if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='--prepare':prepare()
    elif mode=='--worker':
        try:worker()
        except Exception as error:write(RUN/'status.json',dict(state='failed',stage=repr(error),updated=now().isoformat()));raise
    elif mode=='--record':record()
    else:raise ValueError(mode)
