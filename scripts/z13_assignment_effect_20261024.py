"""Readout-action supervision from exact one-step tracker assignments, CPU only."""
from pathlib import Path
from datetime import datetime,timedelta
import os
import subprocess
import sys
import time
import json
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']=os.environ['OPENBLAS_NUM_THREADS']='2'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
RUN=ROOT/'runs/z13_assignment_effect_20261024';W=ROOT/'runs/z13_weekend_20261024'
CACHE=ROOT/'runs/z13_candidate_reserve_20261021';EXTRA=ROOT/'runs/z13_candidate_data104_20261021'
IDENT='z13_assignment_effect_v1';ARMS=('D','C','R1');T0=time.monotonic()


def journal(title,text):
    from scripts import z13_val20_common_20261024 as S
    old=S.IDENT
    try:S.IDENT=IDENT;S.journal(title,text)
    finally:S.IDENT=old


def close_invalid_sampling():
    from scripts.z13_unperturbed_readout_20261024 import flags
    from scripts.z13_hota_deadline_guard import WindowsProcesses
    old=ROOT/'runs/z13_unperturbed_readout_20261024'
    if (old/'stopped_recorded.json').exists():return
    p=read(old/'protocol.json');e=read(old/'controller_result.json')
    assert e['exit_code']==1 and e['all_owned_ended'] and e['worker_ended'] and not e['remaining']
    ps=WindowsProcesses();assert all(ps.creation(int(pid))!=b for pid,b in e['owned'].items())
    a=read(CACHE/'fit/labels.json');b=read(EXTRA/'evaluate/labels.json');v=read(CACHE/'evaluate/labels.json')
    f=np.concatenate([flags(a,p['fit20']),flags(b,p['extra84'])]);vf=flags(v,p['evaluate26'])
    s=read(old/'sampling.json');assert not f.any() and not vf.any() and set(s['clean'])==set(s['mixed'])
    total=pert=0
    for video in p['train104']+p['evaluate26']:
        with np.load(ROOT/f'runs/z13_data/{video}.npz') as z:total+=len(z['ep_pert']);pert+=int(z['ep_pert'].sum())
    done=list(old.glob('*/seed*/done.json'))
    for d in done:assert sha(d.parent/'model.pt')==read(d.parent/'training.json')['model_sha256']
    write(old/'eligibility_audit.json',dict(at=now().isoformat(),original_episodes=total,original_perturbed_episodes=pert,
        selected_training_rows=len(a)+len(b),selected_perturbed_training_rows=int(f.sum()),selected_eval_rows=len(v),selected_perturbed_eval_rows=int(vf.sum()),
        known_rows_each_regime=len(s['clean']),training_sets_identical=True,completed_models_preserved=len(done),
        source_filter='scripts/z13_candidate_reserve_20261021.py:if not l[8];scripts/z13_train.py:labels[8]=ep_pert',
        intervention_not_identifiable=True,agent_requested_stop=True,original_error_preserved=True,all_owned_ended=True))
    cp=ROOT/'experiments/project_history.json';cat=read(cp);card=next(c for c in cat['experiments'] if c['id']=='z13_unperturbed_readout_v1')
    source=(old/'eligibility_audit.json').relative_to(ROOT).as_posix();head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
    fields=dict(summary='Постановка ошибочна:кэш уже исключил сдвинутые эпизоды;остановлено без научного сравнения.',
        obtained=source+':исходных эпизодов19798,со сдвигом9888;в выбранных обучающих/оценочных строках со сдвигом0/0. Обе обучающие выборки содержат одни и те же20778известных строк. Сохранено'+str(len(done))+'готовыхмоделей;controller_result exit1/all_owned_ended после stop.request.',
        interpretation='Агент ошибочно перенёс описание исходного архива на уже отфильтрованный кэш. Два режима не различаются составом строк,поэтому обучение не проверяет гипотезу. Из различия порядка пакетов выводов не делаем. Это ошибка постановки,не отрицательный результат FlyVis.',
        decision='Не продолжать эту постановку и не принимать её модели;веса/логи/источники/exit1 сохранить. До любого нового обучения проверять на конкретных входных строках наличие заявленного различия факторов.')
    for k,value in fields.items():
        if card.get(k) and card[k]!=value:card.setdefault('revisions',[]).append(dict(date=now().strftime('%Y-%m-%d'),commit=head,field=k,old=card[k],reason='Обнаружено отсутствие заявленного различия обучающих строк;явная поправка ошибки агента.'))
        card[k]=value
    card.update(state='inconclusive',done=now().strftime('%Y-%m-%d'));card['acceptance'][0].update(met=None,evidence=source+';comparison not identifiable')
    card['acceptance'][1].update(met=False,evidence=source+';stopped before36models')
    write(cp,cat);write(old/'stopped_recorded.json',dict(at=now().isoformat(),reason='Invalid proposed factor contrast;both row sets already unperturbed',audit_sha256=sha(old/'eligibility_audit.json'),all_owned_ended=True))
    journal('поправка собственной ошибки:кэш уже исключил сдвиг окна',
        '*Ошибка агента.* '+fields['interpretation']+'\n\n*Получено.* '+fields['obtained']+'\n\n*Решение.* '+fields['decision']+'\n\nНи нового научного результата,ни процента улучшения здесь нет. Прежняя остановка не переписана как успех.')


def prepare():
    assert not RUN.exists();close_invalid_sampling()
    old=read(EXTRA/'protocol.json');videos=old['fit']+old['evaluate']+old['evaluation26'];assert len(set(videos))==130
    names=[Path(__file__).relative_to(ROOT).as_posix(),'scripts/z13_assignment_effect_console.ps1',
        'scripts/z13_publish_assignment_effect_20261024.py','tests/test_z13_assignment_effect.py',
        'scripts/z13_bounded_stage_20261022.py','scripts/z13_candidate_reserve_20261021.py','scripts/z13_train.py',
        'scripts/z13_data.py','scripts/fovea_target_30hz.py','scripts/association_log.py','scripts/freeze_bench.py',
        'tracking/xywh_bytetrack.py','tracking/offline_association.py',
        '.venv/Lib/site-packages/ultralytics/trackers/byte_tracker.py',
        '.venv/Lib/site-packages/ultralytics/trackers/bot_sort.py',
        '.venv/Lib/site-packages/ultralytics/trackers/utils/matching.py',
        'runs/z13_candidate_data104_20261021/protocol.json','runs/z13_candidate_reserve_20261021/protocol.json']
    splits=[('fit20',old['fit'],CACHE/'fit'),('extra84',old['evaluate'],EXTRA/'evaluate'),('evaluate26',old['evaluation26'],CACHE/'evaluate')]
    for _,_,folder in splits:
        names.append((folder/'labels.json').relative_to(ROOT).as_posix())
        for arm in ARMS:
            names += [x.relative_to(ROOT).as_posix() for x in (folder/arm).glob('cache_*.npz')]
    for video in videos:
        names += [f'runs/z13_data/{video}.npz',f'runs/train_cache/{video}/detections/raw_video.npz',
            f'runs/train_cache/{video}/alignment.json',f'runs/fly_signatures/{video}/signatures.npz',
            f'data/bdd100k/labels/box_track_20/train/{video}.json']
    hashes={n:sha(ROOT/n) for n in names};auth=read(W/'authorization.json');began=now()
    p=dict(id=IDENT,created=began.isoformat(),device='cpu',budget_seconds=3600,
        save_at=min(began+timedelta(seconds=3600),datetime.fromisoformat(auth['save_at'])).isoformat(),
        hard_stop=min(began+timedelta(seconds=3620),datetime.fromisoformat(auth['hard_stop'])).isoformat(),
        fit20=old['fit'],extra84=old['evaluate'],evaluate26=old['evaluation26'],videos=videos,
        action='Prepare supervision for a replacement FlyVis readout trained on the association consequence of its center action,rather than distance-to-GT-center utility. CPU exact baseline replay only;no new model trained here.',
        method='For every original cached candidate on real annotated query frames,temporarily replace only the lost-track mean center,compute its first-association cost row,and solve the complete unchanged Hungarian assignment with that single row replaced. Restore exact mean before real tracker update. Baseline trajectory remains unchanged. Other tracks participate. Store full assignment alternatives,not independent nearest-detection guesses.',
        truth='Original real annotated frames only,IoU>=0.5 within-class detection-to-GT matching as existing label function. Track identity from its last observed annotated detection as existing data extraction;unknown/freshness/gap cases excluded from semantic outcome. No interpolation. GT only labels saved assignment outcomes;never supplied to baseline tracker/candidate center generation.',
        benefit='Known queried identity:first-stage correct match becomes available where baseline row was incorrect AND total correct assignments over eligible pool rows strictly increases. Harm:own previously correct match lost OR total correct pool count decreases. Else neutral. If no high detection of a known object exists,unmatched is correct for this first stage only. Not final tracking correctness or HOTA.',
        gates='All130videos replay float32 raw outputs exactly against saved baseline;every cached label prediction matches replay mean castfloat32 exactly;all rows bound byvideo,track,frame,identity. First2fixed fit20videos measure walltime;2*mean_seconds*remaining128+180must fit actual remaining original budget,else retain2jobs and stop. Completed videos not repeated.',
        primary='On all26reused training evaluation videos,mean fraction of known eligible queries with at least one D candidate producing net beneficial first-stage assignment >=0.02 and paired-video95%lower>0 against0. If upper<0.02 this candidate set has insufficient measured reserve;otherwise inconclusive. C/R1 reserve and geometry-preserving no-action baseline reported. Old center-useful flags versus benefit/harm table reported. Oracle is diagnostic only;it never becomes a runtime policy.',
        next_action='Only if reserve and reproduction pass,register an equal-budget18model C/D/R1/activity-vsgeometry readout pilot replacing original center-useful training labels by association-beneficial action labels;same features/architecture/BCE/seeds and old-model replay. No HOTA before this gate. No model source is claimed ready before registration.',
        counterargument='One-step baseline-state counterfactuals omit feedback,later low-score/unconfirmed association and long-term identity. Sparse annotations and last observed identity limit semantic labels. Higher oracle reserve need not be learnable. This is an association-aligned readout target hypothesis,not DeepMOT reproduction or a change to YOLO/ByteTrack/MASA.',
        paper=dict(url='https://openaccess.thecvf.com/content_CVPR_2020/html/Xu_How_to_Train_Your_Deep_Multi-Object_Tracker_CVPR_2020_paper.html',support='DeepMOT motivates matching-aware tracker losses. We compute exact frozen one-step action labels,not its differentiable Hungarian network or MOTA loss.'),sha256=hashes)
    write(RUN/'protocol.json',p);write(RUN/'status.json',dict(state='ready',stage='Точное CPU-повторение и последствия поправок',updated=now().isoformat(),done=0,total=130))
    cp=ROOT/'experiments/project_history.json';cat=read(cp)
    assert not any(x['id']==IDENT for x in cat['experiments'])
    cat['experiments'].append(dict(id=IDENT,title='З13: последствия кандидатов считывателя для полного сопоставления',area='flyvis',state='ready',started=now().strftime('%Y-%m-%d'),tags=['fly-problem-P8'],
        question='Есть ли доступные кандидатам D полезные исправления фактического сопоставления,и совпадают ли они с прежними метками центра?',
        hypothesis='Метка улучшения центра может не совпадать с пользой сопоставлению;иной целевой сигнал может лучше обучать отказ.',expected=p['action']+' '+p['method'],
        acceptance=[dict(criterion=p['gates'],met=None),dict(criterion=p['primary'],met=None)],summary='Сначала точный контроль исходного трекера и цена,потом полный диагностический набор;без обучения.',
        obtained='',interpretation='',decision='',cannot_claim=p['counterargument'],metrics=[],commits=[],files=[names[0]],evidence=[dict(kind='file',ref=(RUN/'protocol.json').relative_to(ROOT).as_posix(),claim='Условия реального сопоставления,критерии и цена до расчёта')]))
    write(cp,cat);journal('постановка меток по последствиям сопоставления',
        '*Действие.* '+p['action']+'\n\n*Критерии.* '+p['gates']+' '+p['primary']+'\n\n'
        '*Возражение.* '+p['counterargument']+'\n\nDeepMOT§4,https://openaccess.thecvf.com/content_CVPR_2020/html/Xu_How_to_Train_Your_Deep_Multi-Object_Tracker_CVPR_2020_paper.html '
        'поддерживает согласование обучения с решением трекера;наши точные однокадровые метки — отдельная гипотеза. '
        'Первый2fit20ролика — воспроизведение/цена,а не вывод об эффективности. CPU3600с,безвидео/GPU/ядра/dev/val20подбора/75/загрузок.')


def assignment_vector(cost,threshold):
    from ultralytics.trackers.utils import matching
    matches,_,_=matching.linear_assignment(np.asarray(cost).copy(),thresh=threshold)
    result=np.full(len(cost),-1,np.int32)
    for i,j in matches:result[int(i)]=int(j)
    return result


def action_outcome(baseline,alternative,row):
    """Input booleans are semantic evaluation only,never tracker features."""
    baseline=np.asarray(baseline,bool);alternative=np.asarray(alternative,bool)
    delta=int(alternative.sum())-int(baseline.sum())
    benefit=not bool(baseline[row]) and bool(alternative[row]) and delta>0
    harm=(bool(baseline[row]) and not bool(alternative[row])) or delta<0
    return dict(beneficial=benefit,harmful=harm,correct_count_delta=delta,
        own_before=bool(baseline[row]),own_after=bool(alternative[row]))


def load_queries(p):
    result={v:[] for v in p['videos']};centers={};uid=0
    for split,videos,folder in [('fit20',p['fit20'],CACHE/'fit'),('extra84',p['extra84'],EXTRA/'evaluate'),('evaluate26',p['evaluate26'],CACHE/'evaluate')]:
        labels=read(folder/'labels.json');lookup={}
        for arm in ARMS:
            a={}
            for path in sorted((folder/arm).glob('cache_*.npz')):
                with np.load(path) as z:
                    ep=z['episode'];ts=z['t'];n=z['count'];cs=z['centers']
                for i in range(len(ep)):a[(int(ep[i]),int(ts[i]))]=cs[i,:int(n[i])].copy()
            lookup[arm]=a
        offset=0
        for v in videos:
            with np.load(ROOT/f'runs/z13_data/{v}.npz') as z:
                off=z['ep_offset'];fr=z['fr_frame'];tid=z['ep_tid'];identity=z['ep_identity'];pert=z['ep_pert']
            for l in labels:
                if l['video']!=v:continue
                e=l['episode']-offset;assert not pert[e]
                q=dict(uid=uid,split=split,video=v,frame=int(fr[int(off[e])+4+l['pos']]),tid=int(tid[e]),identity=str(identity[e]),label=l)
                centers[uid]={a:lookup[a][(l['episode'],l['pos']+4)] for a in ARMS};result[v].append(q);uid+=1
            offset+=len(tid)
    return result,centers


def replay_video(video,queries,centers,p):
    from scripts.z13_data import raw_index
    from scripts.association_log import label
    from scripts.output_filter_oracle import _iou
    from scripts.freeze_bench import XYWH_CONFIG
    from tracking.offline_association import _DetSlice,make_tracker_args
    from tracking.xywh_bytetrack import XYWHByteTrack
    from ultralytics.trackers.basetrack import BaseTrack,TrackState
    from datasets.bdd100k_loader import CLASS_NAMES
    byframe={}
    for q in queries:byframe.setdefault(q['frame'],[]).append(q)
    with np.load(ROOT/f'runs/train_cache/{video}/detections/raw_video.npz') as z:det={k:z[k] for k in ('xyxy','conf','cls','frame_idx')}
    with np.load(ROOT/f'runs/fly_signatures/{video}/signatures.npz') as z:n=int(z['frame_count']);expected=z['outputs']
    truth=read(ROOT/f'data/bdd100k/labels/box_track_20/train/{video}.json');vf=[x['video_frame_index'] for x in read(ROOT/f'runs/train_cache/{video}/alignment.json')['matches']]
    assert len(truth)==len(vf)
    R=raw_index(truth,vf,{'pedestrian','rider','car','bus','truck','train','motorcycle','bicycle'})
    ann=sorted(R);neighbors={f:(ann[i-1] if i else None,ann[i+1] if i+1<len(ann) else None) for i,f in enumerate(ann)}
    ctx=dict(frame=-1,events=[]);own={};rows=[];outputs=[]
    class Observe(XYWHByteTrack):
        def _first_association(self,pool,detections,activated,refind):
            f=ctx['frame'];wanted=byframe.get(f,[])
            if wanted:
                costs=self.get_dists(pool,detections);base=assignment_vector(costs,self.args.match_thresh)
                ids=[int(t.track_id) for t in pool];seen=set()
                for q in wanted:
                    assert q['tid'] in ids,(video,f,q['tid'])
                    ri=ids.index(q['tid']);t=pool[ri];assert t.state!=TrackState.Tracked
                    assert np.array_equal(t.mean[:2].astype(np.float32),q['label']['pred']),(video,f,'prediction mismatch')
                    assert (own.get(q['tid'],('',-1))[0] or '')==q['identity'],(video,f,'identity mismatch')
                    event=dict(q=q,row=ri,baseline=base.copy(),alternatives={},ids=ids,
                        causal=[own.get(int(t.track_id),('',-10000)) if (int(t.frame_id)-1)-own.get(int(t.track_id),('',-10000))[1]<=12 else ('',-10000) for t in pool])
                    original=t.mean.copy()
                    try:
                        for arm in ARMS:
                            alts=[]
                            for center in centers[q['uid']][arm]:
                                t.mean[:2]=center;altered=costs.copy();altered[ri]=self.get_dists([t],detections)[0]
                                alts.append(assignment_vector(altered,self.args.match_thresh))
                            event['alternatives'][arm]=alts
                    finally:t.mean[:]=original
                    assert np.array_equal(self.get_dists(pool,detections),costs)
                    ctx['events'].append(event);seen.add(q['uid'])
                assert len(seen)==len(wanted)
            return super()._first_association(pool,detections,activated,refind)
    BaseTrack.reset_id();tr=Observe(make_tracker_args(**XYWH_CONFIG))
    for f in range(n):
        if (RUN/'stop.request').exists() or now()>=datetime.fromisoformat(p['save_at']):raise TimeoutError('Original CPU stage deadline')
        ix=np.flatnonzero(det['frame_idx']==f);boxes,scores,cls=det['xyxy'][ix],det['conf'][ix],det['cls'][ix]
        ctx['frame']=f;ctx['events']=[]
        out=tr.update(_DetSlice(boxes,scores,cls))
        if len(out):outputs.append(np.column_stack([np.full(len(out),f),np.asarray(out)[:,:7]]).astype(np.float32))
        if f in R:
            objects=R[f];labels=label(list(boxes),[CLASS_NAMES[int(c)] for c in cls],objects)
            high=[a for a,s in zip(labels,scores) if s>=tr.args.track_high_thresh]
            objectids={str(a[0]) for a in objects};prev,nxt=neighbors[f]
            gaps={str(a[0]) for a in R.get(prev,[])}&{str(a[0]) for a in R.get(nxt,[])}-objectids
            for event in ctx['events']:
                q=event['q'];eligible=np.asarray([bool(a[0]) and str(a[0]) not in gaps for a in event['causal']])
                if q['label']['kind']>=0:assert eligible[event['row']],(video,f,'Known query lacks eligible causal identity')
                def correctness(assignment):
                    answer=[]
                    for i,((identity,_),j) in enumerate(zip(event['causal'],assignment)):
                        if not eligible[i]:answer.append(False);continue
                        answer.append((j>=0 and high[j]==identity) if identity in high else j<0)
                    return np.asarray(answer,bool)
                baseline=correctness(event['baseline']);rr=dict(uid=q['uid'],split=q['split'],video=video,frame=f,tid=q['tid'],kind=q['label']['kind'],label=q['label']['label'],arms={})
                for arm,alts in event['alternatives'].items():
                    rr['arms'][arm]=[action_outcome(baseline,correctness(a),event['row']) for a in alts] if q['label']['kind']>=0 else [dict(beneficial=None,harmful=None,correct_count_delta=None) for a in alts]
                    for j,outcome in enumerate(rr['arms'][arm]):
                        l=q['label']
                        if l['kind']==1:
                            error=np.linalg.norm(centers[q['uid']][arm][j]-l['gt']);prior=np.linalg.norm(np.asarray(l['pred'])-l['gt'])
                            outcome['old_center_useful']=bool(error<=.25*l['scale'] and error<prior)
                        else:outcome['old_center_useful']=False if l['kind']==0 else None
                rows.append(rr)
            for t in tr.tracked_stracks:
                if t.frame_id==tr.frame_id and len(boxes):own[int(t.track_id)]=(labels[int(np.argmax(_iou([t.xyxy],boxes)[0]))],f)
        if f%100==0:write(RUN/'status.json',dict(state='running',stage='CPU:исходный трекер и последствия кандидатов',updated=now().isoformat(),variant=video,done=f,total=n,elapsed_seconds=round(time.monotonic()-T0,2)))
    actual=np.concatenate(outputs) if outputs else np.zeros((0,8),np.float32)
    assert np.array_equal(actual,expected),(video,'Raw replay mismatch',actual.shape,expected.shape)
    assert len(rows)==len(queries),(video,len(rows),len(queries))
    return rows,dict(frames=n,query_rows=len(rows),raw_replay_exact=True,all_predictions_and_identities_bound=True)


def worker():
    from scripts.z13_localization_labels_20261023 import verify_sources
    from scripts.z13_continue import paired
    p=read(RUN/'protocol.json');verify_sources(p);queries,centers=load_queries(p);times=[]
    for vi,video in enumerate(p['videos']):
        path=RUN/'videos'/(video+'.json')
        if path.exists():continue
        began=time.monotonic();rows,checks=replay_video(video,queries[video],centers,p)
        write(path,dict(rows=rows,checks=checks,seconds=time.monotonic()-began));times.append(time.monotonic()-began)
        if vi==1:
            estimate=2*np.mean(times)*128+180;available=(datetime.fromisoformat(p['save_at'])-now()).total_seconds()
            write(RUN/'price.json',dict(first2_seconds=times,conservative_remaining_seconds=float(estimate),actual_remaining_seconds=available,proceed=estimate<available))
            assert estimate<available,'Full cache does not fit original budget;first2retained'
    values={a:{} for a in ARMS};counts={a:dict(known=0,candidates=0,beneficial=0,harmful=0,
        center_useful_candidates=0,center_useful_beneficial=0,center_useful_harmful=0,center_useful_neutral=0) for a in ARMS}
    for video in p['evaluate26']:
        rows=read(RUN/'videos'/(video+'.json'))['rows'];known=[r for r in rows if r['kind']>=0];assert known
        for arm in ARMS:
            values[arm][video]=float(np.mean([any(x['beneficial'] for x in r['arms'][arm]) for r in known]))
            counts[arm]['known']+=len(known)
            for r in known:
                counts[arm]['candidates']+=len(r['arms'][arm]);counts[arm]['beneficial']+=sum(bool(x['beneficial']) for x in r['arms'][arm]);counts[arm]['harmful']+=sum(bool(x['harmful']) for x in r['arms'][arm])
                for x in r['arms'][arm]:
                    if x['old_center_useful']:
                        counts[arm]['center_useful_candidates']+=1
                        bucket='center_useful_beneficial' if x['beneficial'] else 'center_useful_harmful' if x['harmful'] else 'center_useful_neutral'
                        counts[arm][bucket]+=1
    contrasts={a:paired(v,{k:0. for k in v}) for a,v in values.items()};main=contrasts['D']
    passed=main['n']==26 and main['mean_per_video']>=.02 and main['ci95'][0]>0
    verify_sources(p);write(RUN/'report.json',dict(primary_pass=passed,oracle_beneficial_fraction=contrasts,per_video=values,candidate_counts=counts,
        complete_videos=130,all_raw_replays_exact=True,not_hota=True,limits=p['counterargument'],video_sha256={v:sha(RUN/'videos'/(v+'.json')) for v in p['videos']}))
    write(RUN/'status.json',dict(state='completed',stage='Последствия кандидатов сохранены;не HOTA',updated=now().isoformat(),done=130,total=130))


def record():
    from scripts.z13_hota_deadline_guard import WindowsProcesses
    p=read(RUN/'protocol.json');e=read(RUN/'controller_result.json');assert e['exit_code']==0 and e['all_owned_ended'] and not e['remaining']
    ps=WindowsProcesses();assert all(ps.creation(int(pid))!=b for pid,b in e['owned'].items())
    r=read(RUN/'report.json')
    for video,h in r['video_sha256'].items():assert sha(RUN/'videos'/(video+'.json'))==h
    cp=ROOT/'experiments/project_history.json';cat=read(cp);c=next(x for x in cat['experiments'] if x['id']==IDENT);source=(RUN/'report.json').relative_to(ROOT).as_posix()
    old=c['summary'];head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
    c.setdefault('revisions',[]).append(dict(date=now().strftime('%Y-%m-%d'),commit=head,field='summary',old=old,reason='Полный диагностический набор последствий.'))
    c.update(state='completed_positive' if r['primary_pass'] else 'completed_negative' if r['oracle_beneficial_fraction']['D']['ci95'][1]<.02 else 'inconclusive',done=now().strftime('%Y-%m-%d'),
        summary='Точные однокадровые последствия проверены;оракульный запас '+('достаточен для проверки обучения.' if r['primary_pass'] else 'не прошёл критерий перехода.'),
        obtained=source+' → oracle_beneficial_fraction/candidate_counts: '+json.dumps(dict(oracle=r['oracle_beneficial_fraction'],counts=r['candidate_counts']),ensure_ascii=False),
        interpretation='Метки относятся к полному первому сопоставлению на неизменённой исходной траектории;не учитывают последующие этапы и обратную связь. Это не HOTA и не реализованный прирост.',
        decision='Отдельно зарегистрировать равный пилот обучения18считывателей по меткам последствий.' if r['primary_pass'] else 'Не обучать эту версию по непройденному запасу;сохранить все данные и выбрать следующий узел.')
    c['acceptance'][0].update(met=True,evidence=source+';130videojobs;price.json');c['acceptance'][1].update(met=r['primary_pass'],evidence=source)
    write(cp,cat);journal('итог точных однокадровых последствий','*Получено.* '+c['obtained']+'\n\n*Вывод.* '+c['interpretation']+'\n\n*Решение.* '+c['decision']+'\n\nПроценты HOTA и доля потолка не определены.')
    write(RUN/'finish_recorded.json',dict(at=now().isoformat(),report_sha256=sha(RUN/'report.json'),all_owned_ended=True,primary_pass=r['primary_pass']))


def monitor():
    from scripts.z13_hota_deadline_guard import WindowsProcesses
    ps=WindowsProcesses();write(RUN/'completion.json',dict(pid=os.getpid(),birth=ps.creation(os.getpid()),at=now().isoformat()))
    while not (RUN/'controller_result.json').exists():time.sleep(3)
    e=read(RUN/'controller_result.json')
    if e['exit_code']!=0 or not e['all_owned_ended']:write(RUN/'monitor_failure.json',dict(at=now().isoformat(),result=e));return
    while not (W/'assignment_effect_start_publication.json').exists():
        if now()>=datetime.fromisoformat(read(RUN/'protocol.json')['hard_stop']):raise TimeoutError('Startup checks not complete in original window')
        time.sleep(3)
    record()
    from scripts.z13_publish_assignment_effect_20261024 import finish
    finish()


if __name__=='__main__':
    if sys.argv[1]=='--prepare':prepare()
    elif sys.argv[1]=='--worker':
        try:worker()
        except Exception as error:write(RUN/'status.json',dict(state='failed',stage=repr(error),updated=now().isoformat()));raise
    elif sys.argv[1]=='--monitor':
        try:monitor()
        except Exception as error:write(RUN/'monitor_failure.json',dict(at=now().isoformat(),error=repr(error)));raise
    else:raise ValueError(sys.argv[1])
