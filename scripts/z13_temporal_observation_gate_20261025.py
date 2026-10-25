"""Bounded availability gate for earlier observed regions; no model inference/training."""
from pathlib import Path
from datetime import datetime,timedelta
import os,sys,time,subprocess,json
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts.z13_continue import paired
from scripts.z13_localization_labels_20261023 import verify_sources
RUN=ROOT/'runs/z13_temporal_observation_gate_20261025'
OLD=ROOT/'runs/z13_box_supervision_followup_20261025'
IDENT='z13_temporal_observation_gate_v1'


def journal(title,text):
    from scripts import z13_candidate_reserve_20261021 as m
    saved=m.IDENT
    try:m.IDENT=IDENT;m.journal(title,text)
    finally:m.IDENT=saved


def prepare():
    assert not (RUN/'protocol.json').exists()
    base=read(ROOT/'runs/z13_assignment_effect_20261024/protocol.json')
    files=[Path(__file__).relative_to(ROOT).as_posix(),'tracking/z13_region_assignment.py','tracking/z13_readout.py',
           'runs/z13_local_peak_20261021/geometry.json','scripts/z13_bounded_recovery_controller_20261022.py',
           'runs/z13_box_supervision_followup_20261025/temporal_readiness.json','runs/z13_box_supervision_followup_20261025/strict_action_gate.json']
    for f in ('runs/z13_candidate_reserve_20261021/fit/labels.json','runs/z13_candidate_data104_20261021/evaluate/labels.json','runs/z13_candidate_reserve_20261021/evaluate/labels.json'):files.append(f)
    hashes={f:sha(ROOT/f) for f in files}
    for v in base['videos']:
        for f in (f'runs/z13_data/{v}.npz',f'runs/train_cache/{v}/detections/raw_video.npz'):hashes[f]=base['sha256'][f]
        f=f'runs/z13_box_supervision_followup_20261025/cache/videos/{v}.json';hashes[f]=read(OLD/'cache/report.json')['manifest'][v]
    end=min(now()+timedelta(seconds=600),datetime.fromisoformat(read(ROOT/'runs/z13_weekend_20261024/authorization.json')['save_at']))
    p=dict(id=IDENT,created=now().isoformat(),timezone='Europe/Moscow',device='cpu',budget_seconds=600,save_at=end.isoformat(),hard_stop=(end+timedelta(seconds=20)).isoformat(),stop_request_path=(RUN/'stop.request').relative_to(ROOT).as_posix(),
        fit20=base['fit20'],fit84=base['extra84'],evaluate26=base['evaluate26'],sha256=hashes,
        action='Prepare a distinct temporal FlyVis readout input: real candidate regions in an earlier stored observation of the SAME episode plus unchanged current regions. This gate only chooses and binds inputs; no learner or feature extraction.',
        selection='For each unchanged query, scan stored episode frames strictly earlier than current in reverse chronological order; choose latest frame with supported same-class real high-score regions under unchanged regions() API. Initial four observed exemplar frames are eligible. Never read ep_identity/ground-truth when selecting. Never use future frames or another episode. Preserve current state/history/choices exactly.',
        caveat='Previous prediction center is not reconstructed. Crop midpoint is supplied solely to regions() geometry which is DISCARDED; support and region indices do not depend on prediction. No invented previous prediction enters a model. Observation pixels are already cached fr_hex; activity still needs a separate matched extraction and causal replay. Earlier region is not assumed to be the same true object.',
        criteria='All37296 current UID/episode/frame/track/choice bindings reproduce exactly; every selected reference strictly earlier/same episode; no true identity selection. Candidate resource admission: supported pair count>14252 and paired-video95%lower of oracle-reserve increment over old eligible pairs>0; expanded oracle mean>=.02 and lower>0. Existing model efficacy criteria do not change. If failed do not train this construction.',
        counterargument='A previous visible same-class region is not identity evidence; stale/wrong track IDs and distractors may defeat learning. This is an availability/oracle gate on repeated train26, not predicted model performance/HOTA or proof of temporal causality.',
        next='Passing admits only preparation of matched D/C/R1 temporal features and their replay/cost gate. No automatic model training, no thresholds chosen on26, no automatic HOTA.',
        literature='KeepTrack,ICCV2021 https://openaccess.thecvf.com/content/ICCV2021/html/Mayer_Learning_Target_Candidate_Association_To_Keep_Track_of_What_Not_ICCV_2021_paper.html motivates learned cross-frame candidate identities; this metadata gate is not its implementation.')
    write(RUN/'protocol.json',p);write(RUN/'status.json',dict(state='ready',updated=now().isoformat(),done=0,total=130,stage='Доступность прежних наблюдений и областей'))
    cat=read(ROOT/'experiments/project_history.json');assert not any(c['id']==IDENT for c in cat['experiments'])
    cat['experiments'].append(dict(id=IDENT,title='Готовность прежних наблюдений для межкадрового считывания FlyVis',area='flyvis',state='running',started=now().strftime('%Y-%m-%d'),question='Можно ли расширить временные пары за пределы прежних14252без будущих кадров/истинной личности?',hypothesis='В сохранённых эпизодах есть более ранние наблюдения с реальными областями, даже если прежнего запроса к мухе не было; это не гипотеза о качестве будущей модели.',expected=p['action'],acceptance=[dict(criterion=p['selection'],met=None),dict(criterion=p['criteria'],met=None)],summary='Дешёвая проверка входов, без обучения иGPU.',obtained='',interpretation='',decision='',cannot_claim=p['counterargument'],tags=['fly-problem-P8'],evidence=[dict(kind='file',ref=(RUN/'protocol.json').relative_to(ROOT).as_posix(),claim='Правила и критерии до проверки130видео')]))
    write(ROOT/'experiments/project_history.json',cat)
    journal('подготовка отдельной временной ветви: проверка доступности','*Действие.* '+p['action']+'\n\n*Критерии до расчёта.* '+p['criteria']+'\n\n*Наблюдаемый выбор.* '+p['selection']+'\n\n*Ограничение.* '+p['caveat']+' '+p['counterargument']+'\n\n*Бюджет.* Только чтение метаданных/геометрии CPU600с+20с, независимый контроллер, без обучения/видео/GPU. Пространственный модуль остаётся временно отложенным. '+p['literature'])


def worker():
    from tracking.z13_region_assignment import regions
    class Geometry:
        def __init__(self):
            g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');self.points=np.asarray(g['points'],float);self.eye_size=g['eye_size'];self.spacing=g['spacing']
        def nearest(self,p):return int(np.argmin(((self.points-np.asarray(p))**2).sum(1)))
    geom=Geometry();p=read(RUN/'protocol.json');started=time.monotonic()
    # Large immutable inputs are verified once per video below, under the controller.
    for n,h in p['sha256'].items():
        if not n.endswith('.npz') and '/cache/videos/' not in n:assert sha(ROOT/n)==h,n
    prior=read(OLD/'temporal_readiness.json');uid=0;complete=[];per_old={};per_new={};total_pairs=0;hashes={}
    sets=[('fit20',p['fit20'],ROOT/'runs/z13_candidate_reserve_20261021/fit/labels.json'),('fit84',p['fit84'],ROOT/'runs/z13_candidate_data104_20261021/evaluate/labels.json'),('train26',p['evaluate26'],ROOT/'runs/z13_candidate_reserve_20261021/evaluate/labels.json')]
    for split,videos,label_path in sets:
        labels=read(label_path);offset=0
        for v in videos:
            if now()>=datetime.fromisoformat(p['save_at']) or (RUN/'stop.request').exists():raise TimeoutError('Original temporal metadata budget')
            write(RUN/'status.json',dict(state='running',updated=now().isoformat(),stage='Проверка прежних наблюдений',video=v,done=len(complete),total=130))
            target=RUN/'videos'/(v+'.json');datafile=ROOT/f'runs/z13_data/{v}.npz';detfile=ROOT/f'runs/train_cache/{v}/detections/raw_video.npz';cache=OLD/'cache/videos'/(v+'.json')
            for path in (datafile,detfile,cache):
                key=path.relative_to(ROOT).as_posix();assert sha(path)==p['sha256'][key],key
            with np.load(datafile) as z:d={k:z[k] for k in ('ep_offset','ep_tid','ep_pert','fr_frame','fr_crop')}
            entries=read(cache)['entries']
            with np.load(detfile) as z:det={k:z[k] for k in ('xyxy','conf','cls','frame_idx')}
            support_cache={};rows=[];denominator=numerator=old_num=0
            def supported(frame,crop,category):
                key=(int(frame),tuple(int(x) for x in crop),int(category))
                if key not in support_cache:
                    mask=(det['frame_idx']==frame)&(det['conf']>=.25);ids=np.flatnonzero(mask)
                    choices,_,_=regions(geom,crop,(np.asarray(crop[:2])+crop[2:])/2,det['xyxy'][ids],det['conf'][ids],det['cls'][ids],category)
                    support_cache[key]=(choices,ids)
                return support_cache[key]
            for label in labels:
                if label['video']!=v:continue
                e=int(label['episode'])-offset;assert not d['ep_pert'][e]
                current=int(d['ep_offset'][e])+4+int(label['pos']);frame=int(d['fr_frame'][current]);en=entries[str(uid)];category=en['track_class']
                choices,_=supported(frame,d['fr_crop'][current],category);assert choices==en['choices']
                previous=None
                for index in range(current-1,int(d['ep_offset'][e])-1,-1):
                    earlier=int(d['fr_frame'][index])
                    if earlier>=frame:continue
                    old_choices,raw_ids=supported(earlier,d['fr_crop'][index],category)
                    if len(old_choices)>1:
                        previous=dict(frame=earlier,frame_array_index=index,crop=d['fr_crop'][index].tolist(),raw_detection_indices=raw_ids[np.asarray(old_choices[1:],int)].tolist());break
                eligible=len(choices)>1 and previous is not None
                if previous is not None:assert previous['frame']<frame and int(d['ep_offset'][e])<=previous['frame_array_index']<current
                known=en['kind']>=0;helpful=bool(known and any(en['good'][1:]));denominator+=known;numerator+=int(eligible and helpful)
                old_eligible=bool(prior['previous_query'].get(str(uid),{}).get('eligible',False));old_num+=int(old_eligible and helpful)
                rows.append(dict(uid=uid,video=v,episode_local=e,track_id=int(d['ep_tid'][e]),current_frame=frame,current_array_index=current,current_choices=choices,previous=previous,eligible=eligible,old_pair_eligible=old_eligible))
                uid+=1;total_pairs+=eligible
            assert {str(x['uid']) for x in rows}==set(entries) and len(rows)==len(entries)
            write(target,dict(complete=True,rows=rows,known=denominator,oracle_helpful=numerator,old_oracle_helpful=old_num));hashes[target.relative_to(ROOT).as_posix()]=sha(target)
            per_new[v]=numerator/denominator;per_old[v]=old_num/denominator;complete.append(v);offset+=len(d['ep_tid'])
    assert uid==prior['counts']['rows']==37296
    zeros={v:0. for v in p['evaluate26']};new={v:per_new[v] for v in zeros};old={v:per_old[v] for v in zeros}
    old_result=paired(old,zeros);assert old_result==prior['eligible_oracle'],(old_result,prior['eligible_oracle'])
    new_result=paired(new,zeros);difference=paired(new,old)
    admitted=total_pairs>14252 and new_result['mean_per_video']>=.02 and new_result['ci95'][0]>0 and difference['ci95'][0]>0
    write(RUN/'report.json',dict(complete=True,rows=uid,supported_pairs=total_pairs,previous_supported_pairs=14252,old_oracle=old_result,expanded_oracle=new_result,reserve_increment=difference,resource_admission=admitted,manifest=hashes,all_current_choices_exact=True,all_selected_observations_strictly_past=True,selection_without_true_identity=True,matched_arm_features_not_yet_extracted=True,model_trained=False,hota_measured=False,seconds=time.monotonic()-started))
    write(RUN/'status.json',dict(state='completed',updated=now().isoformat(),done=130,total=130,resource_admission=admitted))


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--worker' in sys.argv:worker()
