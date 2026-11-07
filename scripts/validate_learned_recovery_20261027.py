"""Полный причинный вывод фиксированных голов на прежних40val; исходный трекер не меняется."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
from datetime import datetime,timezone
from types import SimpleNamespace
import copy,json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.recovery_research_service_20261027 import RUN as BASE,read,write,sha,matching
RUN=BASE/'learned_validation'

def prepare():
    assert not RUN.exists();frozen=read(BASE/'frozen_for_validation.json');assert len(frozen['models'])==6
    done=read(BASE/'dev_threshold/controller_result.json');assert done['exit_code']==0 and done['all_owned_ended']
    assert not any(matching(k,v) for k,v in done['owned'].items())
    current=read(BASE/'error_budget/recovery1/protocol.json');sources=[];old=ROOT/'runs/b123_20261027'
    for row in current['sources']:
        v=row['video'];first=row['group']=='development20'
        cache=ROOT/'runs/z13_val20_matched_20261024/cache'/v if first else old/'supported_transfer/recovery2/cache'/v
        detection=cache/'detections/raw_video.npz' if first else cache/'raw_video.npz'
        arrays=ROOT/'runs/z13_val20_matched_20261024/raw/xywh/raw'/(v+'.npz') if first else cache/'tracker_arrays.npz'
        rule=old/'detection_supported_recovery/kalman_support'/(v+'.json') if first else old/'supported_transfer/recovery2/supported'/(v+'.json')
        rule_report=read(old/'detection_supported_recovery/report.json') if first else read(old/'supported_transfer/recovery2/report.json')
        rule_metrics=rule_report['per_video']['kalman_support' if first else 'supported'][v]
        sources.append(dict(**row,detections=detection.relative_to(ROOT).as_posix(),alignment=(cache/'alignment.json').relative_to(ROOT).as_posix(),
            video_path=f'data/bdd100k/videos/val/{v}.mov',raw_array=arrays.relative_to(ROOT).as_posix(),raw_array_key='outputs' if first else 'baseline',
            rule=rule.relative_to(ROOT).as_posix(),rule_metrics=rule_metrics))
    bound=['scripts/validate_learned_recovery_20261027.py','tracking/learned_recovery_admission.py','tracking/recovery_admission_features.py','tracking/detection_supported_kalman.py',
        'tracking/xywh_bytetrack.py','tracking/busca_runtime.py','weights/busca_official_20261027/model_busca.pth','runs/recovery_research_20261027/frozen_for_validation.json',
        'runs/recovery_research_20261027/admission_learning/protocol.json']
    for item in sources:bound += [item[k] for k in ['baseline','detections','alignment','video_path','raw_array','rule']]
    for m in frozen['models']:
        for key,hashkey in [('checkpoint','checkpoint_sha256'),('normalization','normalization_sha256')]:
            path=BASE/'admission_learning'/m[key];assert sha(path)==m[hashkey];bound.append(path.relative_to(ROOT).as_posix())
    p=dict(parent_experiment='recovery_admission_learning_v1',created=datetime.now(timezone.utc).isoformat(),budget_seconds=3600,
        author_stop=read(BASE/'authorization.json')['save_at'],device='cuda',sources=sources,groups=current['groups'],
        config=read(BASE/'admission_dataset/protocol.json')['config'],models=frozen['models'],
        action='ТотжеYOLOкэш30Гц/XYWH,памятьтолькопореальнымпрошлымназначениям,кандидатыпрежнегоправила. '
               'Сохранённыевеса/нормировки/пороги6моделейнеизменны. ПричинныеROIпроходяттакойжеJPEG95roundtripвпамяти,какприобучении. '
               'ReIDимеетзамороженнуюBN;пакетыразныхкадровматематическинеобмениваютсяданными,веса/статистикинеобновляются. '
               'Исходныеназначения/состояниетрекеранеменяются. РазметкаGTтолькоуоценщика;шаблонывыдачиимеюттолькометаданныезаранеезаданныхзапросов.',
        acceptance=['Все40,точныеисходныеraw-массивыинаблюдаемыеJSONбазисаиправила;однидетекции/кандидаты/входы',
                    'НикакихGTвходов/будущихнаблюдений/переподборапороговиливесов,все3сидаобоихвариантов',
                    'HOTA/AssA/IDF1/FP/FN/IDSw,парные95%интервалыповидео;положительныенижниеграницыкбазеикправилувовсех3сидахдляподтвержденияпользыварианта;'
                    'длявкладавнешнеговидатакжеположительныйинтервалкчисловомуконтролюпосопоставимымсидам. Частныеположительныесравнениянеподменяютобщийвывод'],
        cost_gate='Первое ПОЛНОЕвидео:декодирование/исходныйтрекер/признаки/6голов/оценка;2*стоимость*39+120с<остаток3600с. Приfalseсохранитьпервое,непродлеватьи неповторятьготовое.',
        limitation='Историческийvalсобщейисториейпроекта,не новыйнезависимыйитоговыйтест. Полезностьвсегопроекта,неядраFlyVis. '
                   'Проверяетсятажеузкаяветвьдопуска,неисправлениесущественногорезервагеометрииилиосновногосопоставления. '
                   'Пакетныйэквивалентпричинныхвыходовнеявляетсязамеромонлайнзадержки/30Гцпроизводительности.',
        sha256={s:sha(ROOT/s) for s in set(bound)})
    write(RUN/'protocol.json',p)
    write(BASE/'ready_jobs/070_learned_validation.json',dict(name='070_learned_validation',folder=RUN.relative_to(ROOT).as_posix(),script='scripts/validate_learned_recovery_20261027.py',
        script_sha256=sha(__file__),protocol_sha256=sha(RUN/'protocol.json')))
    from scripts.b123_safe_journal_20261027 import journal
    journal('recovery_admission_learning_v1','пороги и веса зафиксированы до полного val-инференса',
        '*Постановка.* '+p['action']+'\n\n*Приёмка уточнена до val.* '+p['acceptance'][2]+' '
        'Уточнение раскрывает «с разбросом сидов» исходного критерия: положительныйнижнийпределтребуетсявовсехтрёх,нельзя выбратьлучшийсидпотесту. '
        'Прежнийкритерийвкарточкенеудаляется. '+p['cost_gate']+' '+p['limitation']+' '
        'Источник runs/recovery_research_20261027/learned_validation/protocol.json,родитель recovery_admission_learning_v1. '
        'НикакойновойHOTAпоvalнаэтомшагеещёнет. Послеэтойоднойполнойпроверкинеуходитьвбесконечныетюнингиузкогодопуска;следующийкрупныйузелпорезерву.')
    print('Frozen full validation prepared; heads and thresholds cannot change on val')

def run():
    import cv2,numpy as np,torch
    from tracking.xywh_bytetrack import XYWHByteTrack
    from tracking.offline_association import _DetSlice
    from tracking.detection_supported_kalman import DetectionSupportedKalmanOutput
    from tracking.recovery_admission_features import CausalAdmissionMemory,FEATURE_NAMES
    from tracking.learned_recovery_admission import AdmissionModel
    from tracking.busca_runtime import load_model
    from tracking.gpu_lock import GpuLock
    from scripts.run_bdd_video_rate_experiment import CLASS_NAMES,load_detection_cache,detections_for_frame,scalabel_frame
    from evaluation.hota_eval import evaluate_hota
    from scripts.z13_continue import paired
    p=read(RUN/'protocol.json');end=datetime.fromisoformat(read(RUN/'execution_deadline.json')['deadline']).timestamp();started=time.monotonic()
    for name,h in p['sha256'].items():assert sha(ROOT/name)==h,name
    torch.set_num_threads(2);torch.manual_seed(27);torch.cuda.manual_seed_all(27)
    def status(**kw):
        if time.time()>end-5 or (RUN/'stop.request').exists():raise TimeoutError('Full validation budget')
        write(RUN/'status.json',dict(state='running',updated=datetime.now(timezone.utc).isoformat(),**kw),required=False)
    per={'baseline':{},'rule':{}};proof={};timing={};specs={}
    with GpuLock('Local\\DissertationSingleGpu'):
        status(stage='Загрузка фиксированных моделей');model,coverage=load_model('cuda');encoder=model.reid_encoder;encoder.use_domain_adaptation=False;encoder.model.eval()
        buffers={n:x.clone() for n,x in encoder.model.named_buffers()}
        for spec in p['models']:
            name=Path(spec['checkpoint']).stem;specs[name]=spec;per[name]={}
        heads={name:AdmissionModel(BASE/'admission_learning'/m['checkpoint'],BASE/'admission_learning'/m['normalization'],'cuda') for name,m in specs.items()}
        for index,source in enumerate(p['sources']):
            began=time.monotonic();video=source['video'];expected=read(ROOT/source['baseline']);expected_rule=read(ROOT/source['rule'])
            align=read(ROOT/source['alignment'])['matches'];assert len(align)==len(expected)
            anchors={int(a['video_frame_index']):i for i,a in enumerate(align)};assert len(anchors)==len(align)
            dets=load_detection_cache(ROOT/source['detections']);tracker=XYWHByteTrack(SimpleNamespace(**p['config']));tracker.reset_id()
            branch=DetectionSupportedKalmanOutput(CLASS_NAMES,enabled=True);memory=CausalAdmissionMemory();cap=cv2.VideoCapture(str(ROOT/source['video_path']));assert cap.isOpened()
            base_rows=[];rule_rows=[];raw=[];xs=[];images=[];records=[]
            for frame in range(int(dets['frame_count'])):
                if frame%100==0:status(stage='Причинный трекер и области',video=video,index=index,total=40,frame=frame)
                ok,image=cap.read();assert ok,(video,frame)
                xyxy,conf,classes=detections_for_frame(dets,frame);base=np.asarray(tracker.update(_DetSlice(xyxy,conf,classes)),np.float32).reshape(-1,8)
                if len(base):raw.append(np.column_stack([np.full(len(base),frame),base[:,:7]]).astype(np.float32))
                if frame in anchors:
                    ref=anchors[frame];template={k:expected[ref][k] for k in ['name','frameIndex']}
                    base_row=scalabel_frame(video,template,base);assert base_row==expected[ref]
                    extended=branch.output(tracker,base,xyxy,conf,classes);rule_row=scalabel_frame(video,template,extended);assert rule_row==expected_rule[ref]
                    labels=[branch._label(r) for r in base];tracks={t.track_id:t for t in tracker.lost_stracks}
                    for c in [branch._label(r) for r in extended[len(base):]]:
                        x,current,past,past_frame=memory.describe(frame,c,tracks[int(c['id'])],labels,xyxy,conf,classes,image)
                        assert past_frame<frame;xs.append(x);records.append(dict(reference=ref,id=c['id'],frame=frame,past_frame=past_frame))
                        for crop in [past,current]:
                            ok,jpeg=cv2.imencode('.jpg',crop,[cv2.IMWRITE_JPEG_QUALITY,95]);assert ok;images.append(cv2.imdecode(jpeg,cv2.IMREAD_COLOR))
                    base_rows.append(base_row);rule_rows.append(rule_row)
                memory.remember(frame,base,image,{t.track_id for t in tracker.tracked_stracks+tracker.lost_stracks})
            cap.release();actual=np.concatenate(raw) if raw else np.zeros((0,8),np.float32)
            original=np.load(ROOT/source['raw_array'])[source['raw_array_key']];assert np.array_equal(actual,original),(video,'raw baseline')
            embeddings=[]
            with torch.inference_mode():
                for offset in range(0,len(images),16):
                    status(stage='Фиксированные признаки областей',video=video,index=index,images_done=offset,images_total=len(images))
                    array=np.stack(images[offset:offset+16]).astype(np.float32)/255
                    array=(array-np.array([.406,.456,.485],np.float32))/np.array([.225,.224,.299],np.float32)
                    tensor=torch.from_numpy(array[...,::-1].copy()).permute(0,3,1,2).to('cuda');_,z=encoder(tensor);assert torch.isfinite(z).all();embeddings.append(z.cpu().numpy())
            e=np.concatenate(embeddings).reshape(-1,2,512) if embeddings else np.zeros((0,2,512),np.float32)
            x=np.asarray(xs,np.float32).reshape(-1,len(FEATURE_NAMES));assert len(e)==len(x)==len(records)
            folder=RUN/'features';folder.mkdir(exist_ok=True);np.savez_compressed(folder/(video+'.npz'),X=x,E=e)
            write(folder/(video+'_records.json'),records)
            write(RUN/'baseline'/(video+'.json'),base_rows);write(RUN/'rule'/(video+'.json'),rule_rows)
            per['baseline'][video]=source['baseline_metrics'];per['rule'][video]=source['rule_metrics']
            for name,head in heads.items():
                status(stage='Фиксированная голова и HOTA',video=video,index=index,model=name)
                probs=head.probabilities(x,e);assert np.isfinite(probs).all();np.save(folder/(video+'_'+name+'_probabilities.npy'),probs)
                allow={}
                for record,prob in zip(records,probs):
                    if prob>=specs[name]['threshold']:allow.setdefault(record['reference'],set()).add(record['id'])
                rows=[]
                for ref,(b,s) in enumerate(zip(base_rows,rule_rows)):
                    out=copy.deepcopy(b);out['labels'] += [copy.deepcopy(c) for c in s['labels'][len(b['labels']):] if c['id'] in allow.get(ref,set())]
                    assert out['labels'][:len(b['labels'])]==b['labels'] and len({c['id'] for c in out['labels']})==len(out['labels']);rows.append(out)
                write(RUN/name/(video+'.json'),rows);per[name][video]=evaluate_hota(RUN/name,[video],split='val')
            assert all(torch.equal(buffers[n],b) for n,b in encoder.model.named_buffers())
            proof[video]=dict(raw_arrays_exact=True,baseline_json_exact=True,rule_json_exact=True,strict_past=True,candidates=len(records),features_sha256=sha(folder/(video+'.npz')))
            timing[video]=time.monotonic()-began;write(RUN/'per_video.json',per);write(RUN/'completed_videos.json',dict(timing=timing,proof=proof))
            if index==0:
                cost=2*timing[video]*39+120;available=end-time.time();write(RUN/'price.json',dict(first_full_video_seconds=timing[video],conservative_remaining_seconds=cost,available_seconds=available,proceed=cost<available));assert cost<available
        groups=dict(p['groups']);groups['all40']=[s['video'] for s in p['sources']];contrasts={}
        for group,videos in groups.items():
            contrasts[group]={}
            for name in heads:
                for control in ['baseline','rule']:
                    contrasts[group][name+'_vs_'+control]=paired({v:per[name][v]['HOTA'] for v in videos},{v:per[control][v]['HOTA'] for v in videos})
            for seed in [42,43,44]:contrasts[group]['appearance_vs_numeric_seed'+str(seed)]=paired({v:per['appearance_seed'+str(seed)][v]['HOTA'] for v in videos},{v:per['numeric_seed'+str(seed)][v]['HOTA'] for v in videos})
        pooled={}
        for name in per:status(stage='Общая HOTA',model=name);pooled[name]=evaluate_hota(RUN/name,groups['all40'],split='val')
        write(RUN/'report.json',dict(complete=True,per_video=per,pooled=pooled,paired=contrasts,proof=proof,timing=timing,
            fixed_models=p['models'],weights_changed_on_val=False,ground_truth_used_by_predictor=False,flyvis_core_used=False,
            batchnorm_buffers_unchanged=True,strict_weight_coverage=coverage,seconds=time.monotonic()-started,limitation=p['limitation']))
    write(RUN/'status.json',dict(state='completed',updated=datetime.now(timezone.utc).isoformat()))

if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    else:
        try:run()
        except Exception as e:write(RUN/'status.json',dict(state='failed',error=repr(e),updated=datetime.now(timezone.utc).isoformat()));raise
