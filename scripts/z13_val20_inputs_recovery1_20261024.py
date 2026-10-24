"""Recover lossy-thumbnail alignment conflicts using original pixel data only."""
from pathlib import Path
from datetime import datetime
import os,sys,time,subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import z13_val20_inputs_20261024 as I
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
OLD=I.STAGE;REC=I.RUN/'inputs_recovery1'


def prepare():
    from scripts.z13_hota_deadline_guard import WindowsProcesses
    from scripts.z13_localization_labels_20261023 import verify_sources
    assert not REC.exists();p=read(OLD/'protocol.json');verify_sources(p);e=read(OLD/'controller_result.json');c=read(OLD/'controller.json');ps=WindowsProcesses()
    assert e['exit_code']==1 and e['all_owned_ended'] and not e['remaining'] and ps.creation(c['pid'])!=c['birth']
    d=read(OLD/'alignment_native_audit.json')['rows'];assert [x['native_best'] for x in d]==[395,401] and [x['compact_best'] for x in d]==[402,401]
    remaining=int((datetime.fromisoformat(c['save_at'])-now()).total_seconds());assert remaining>600
    names=['scripts/z13_val20_inputs_recovery1_20261024.py','scripts/z13_val20_alignment_audit_20261024.py','scripts/z13_val20_alignment_native_audit_20261024.py','scripts/z13_val20_queue_20261024.py','scripts/z13_val20_raw_20261024.py','scripts/z13_val20_raw_console.ps1','scripts/z13_val20_inputs_console.ps1']
    names += [str((OLD/n).relative_to(ROOT).as_posix()) for n in ('alignment_failure_audit.json','alignment_native_audit.json','controller_result.json')]
    p={**p,'created':now().isoformat(),'budget_seconds':remaining,'save_at':c['save_at'],'hard_stop':c['hard_stop'],'stop_request_path':(REC/'stop.request').relative_to(ROOT).as_posix(),'sha256':{**p['sha256'],**{n:sha(ROOT/n) for n in names}},'alignment_service_recovery':'Original160x90 independent matching sometimes reverses adjacent references due to information loss. Original native pixels prove395,401 instead402,401 within unchanged±24. Original alignment attempted first. Only members of non-increasing adjacent pairs are rechecked at native resolution in their same original windows with the same mean absolute pixel error. Require a unique best candidate and strict monotonicity of the entire result after one repair; otherwise stop. No video replacement, enlarged window, relaxed threshold, GT tracking outcomes, or new reference images. Oldaccepted mapping/finished inputs remain identical. All original sources and failure logs retained.'}
    write(REC/'protocol.json',p)
    for marker in (OLD/'jobs').glob('*.json'):
        m=read(marker)
        for n,h in m['outputs'].items():assert sha(ROOT/n)==h
        write(REC/'jobs'/marker.name,m)
    I.journal('проверено восстановление привязки кадров без расширения окна','Сбой второгоиз20val сохранён:компактные160×90изображения дают обратные402,401. Независимаяпроверка техжеисходныхпикселейвтомжеокне±24даёт уникальные395,401. Новыйслужебныйадаптер сначалаиспользуетпрежнююфункцию;толькочленыконфликтующейпары перепроверяетвисходномразрешении. Тажеабсолютнаяошибка,окно24,уникальныйминимум истрогаямонотонность;послеоднойпоправкиприконфликтесноваостановка. Данные/веса/пороги/фиксированные20неизменны;старыйкод иошибка сохранены. Контрольдетектора ипервыйновыйроликпровереныипропускаются. Исходныйбюджет3600сот04:34:40непродлевается;дляостаткаценаужесохранённогопервогороликанедостаточна,ценавосстановленноговторогоприменяетсякоставшимся18.')


def align(video,images,output,radius):
    import cv2,numpy as np
    from scripts import run_bdd_video_rate_experiment as V
    from tracking.video_alignment import decode_compact_video,load_compact_images
    try:return ORIGINAL(video,images,output,radius)
    except RuntimeError as e:
        if 'не монотонно' not in str(e):raise
    frames,meta=decode_compact_video(video);refs=load_compact_images(images);rows=[]
    for i,(name,ref) in enumerate(zip(images,refs)):
        expected=round(i*(len(frames)-1)/max(1,len(refs)-1));lo=max(0,expected-radius);hi=min(len(frames),expected+radius+1);cost=np.array([cv2.absdiff(ref,x).mean() for x in frames[lo:hi]]);order=np.argsort(cost);j=int(order[0]);rows.append(dict(reference_index=i,reference_name=name.name,video_frame_index=lo+j,mean_absolute_error=float(cost[j]),second_best_error=float(cost[order[1]])))
    bad=[i for i in range(1,len(rows)) if rows[i]['video_frame_index']<=rows[i-1]['video_frame_index']];members=sorted(set(j for i in bad for j in (i-1,i)));assert members
    native={i:cv2.imread(str(images[i])) for i in members};costs={i:[] for i in members};cap=cv2.VideoCapture(str(video));f=0
    try:
        while True:
            ok,frame=cap.read()
            if not ok:break
            for i in members:
                expected=round(i*(len(frames)-1)/max(1,len(refs)-1))
                if abs(f-expected)<=radius:
                    assert native[i].shape==frame.shape;costs[i].append((float(cv2.absdiff(native[i],frame).mean()),f))
            f+=1
    finally:cap.release()
    assert f==meta['decoded_frame_count'];repairs=[]
    for i in members:
        values=sorted(costs[i]);assert values[0][0]<values[1][0],'Native matching remains ambiguous';before=dict(rows[i]);rows[i].update(video_frame_index=values[0][1],mean_absolute_error=values[0][0],second_best_error=values[1][0]);repairs.append(dict(before=before,after=rows[i]))
    assert all(rows[i]['video_frame_index']>rows[i-1]['video_frame_index'] for i in range(1,len(rows))),'Native conflict persists; no relaxation'
    errors=np.array([x['mean_absolute_error'] for x in rows]);steps=np.diff([x['video_frame_index'] for x in rows]);audit=dict(video=meta,reference_frames=len(images),search_radius=radius,matches=rows,maximum_match_mae=float(errors.max()),median_match_mae=float(np.median(errors)),raw_step_counts={str(int(s)):int((steps==s).sum()) for s in np.unique(steps)},service_native_conflicts=repairs,warning='Mixed compact/native errors are provenance only, not a single-resolution accuracy distribution.')
    V.save_json(output,audit);write(REC/'alignment_repairs'/((Path(video).stem)+'.json'),dict(video=Path(video).stem,repairs=repairs,same_radius=radius,all_strictly_increasing=True));return rows,meta


def bind():
    global ORIGINAL
    from scripts import run_bdd_video_rate_experiment as V
    I.STAGE=REC;I.T0=time.monotonic();ORIGINAL=V.build_alignment;V.build_alignment=align


def child(video,probe=False):
    marker=REC/'jobs'/('probe.json' if probe else video+'.json')
    if marker.exists():
        m=read(marker)
        for n,h in m['outputs'].items():assert sha(ROOT/n)==h
        return m
    log=REC/'logs'/('probe' if probe else video);log.parent.mkdir(parents=True,exist_ok=True);began=time.monotonic()
    with log.with_suffix('.stdout.log').open('w',encoding='utf-8') as out,log.with_suffix('.stderr.log').open('w',encoding='utf-8') as err:
        p=subprocess.Popen([str(ROOT/'.venv/Scripts/python.exe'),'-u',str(Path(__file__)),'--job',video,*(['--probe'] if probe else [])],cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.IDLE_PRIORITY_CLASS|subprocess.CREATE_NO_WINDOW)
        while p.poll() is None:I.check();time.sleep(.5)
        assert p.returncode==0,('Input recovery child failed',video,p.returncode)
    m=read(marker);m['wall_seconds_with_imports']=time.monotonic()-began;write(marker,m);return m


def worker():
    from scripts.z13_localization_labels_20261023 import verify_sources
    p=read(REC/'protocol.json');verify_sources(p);child(p['probe_video'],True);first_new=True
    for i,v in enumerate(p['videos']):
        already=(REC/'jobs'/(v+'.json')).exists();m=child(v)
        if not already and first_new:
            estimate=2*m['wall_seconds_with_imports']*(19-i)+120;available=p['budget_seconds']-(time.monotonic()-I.T0);write(REC/'price.json',dict(measured_video=v,conservative_remaining_seconds=estimate,available_seconds=available,proceed=estimate<available));assert estimate<available,'Recovery input price';first_new=False
    verify_sources(p);jobs=list((REC/'jobs').glob('*.json'));assert len(jobs)==21
    write(REC/'report.json',dict(videos=p['videos'],jobs=21,total_frames=sum(p['frames'][v] for v in p['videos']),detector_replay_exact=True,metadata_and_detection_hashes={q.name:read(q)['outputs'] for q in jobs},no_hota_or_training=True,limitations=p['limitations']));write(REC/'status.json',dict(state='completed',stage='20входныхнаборов готовы',updated=now().isoformat(),jobs_done=21,jobs_total=21))


def queue():
    from scripts import z13_val20_queue_20261024 as Q
    I.STAGE=REC;Q.I.STAGE=REC;Q.R.I.STAGE=REC;original=Q.update
    def revised(stage,ident,ok,obtained,interpretation,decision):return original(stage,ident,ok,obtained.replace('z13_val20_matched_20261024/inputs/','z13_val20_matched_20261024/inputs_recovery1/'),interpretation,decision)
    Q.update=revised;Q.main()


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--job' in sys.argv:bind();I.job(sys.argv[2],'--probe' in sys.argv)
    elif '--worker' in sys.argv:
        bind()
        try:worker()
        except Exception as e:write(REC/'status.json',dict(state='failed',stage=repr(e),updated=now().isoformat()));raise
    elif '--queue' in sys.argv:queue()
    else:raise RuntimeError('Explicit recovery mode required')
