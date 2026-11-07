"""Продолжить только оставшиеся видео после проверенного уточнения привязки."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,sys,math,time,shutil
import psutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));PARENT=ROOT/'runs/b123_20261027/supported_transfer';OLD=PARENT/'recovery1';RUN=PARENT/'recovery2';ORIGINAL=ROOT/'scripts/b123_supported_transfer_20261027.py'
from scripts.b123_supported_transfer_recovery1_20261027 import write
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def prepare():
    assert not RUN.exists();end=read(OLD/'controller_result.json');assert end['exit_code']!=0 and end['all_owned_ended']
    for pid,birth in end['owned'].items():
        if psutil.pid_exists(int(pid)):assert abs(psutil.Process(int(pid)).create_time()-birth)>.01
    assert 'не монотонно' in (OLD/'stderr.log').read_text(encoding='utf-8')
    done=read(OLD/'completed_videos.json');assert len(done['videos'])==10
    p=read(OLD/'protocol.json');assert done['videos']==p['videos'][:10]
    deadline=read(OLD/'execution_deadline.json')['deadline'];remaining=datetime.fromisoformat(deadline).timestamp()-time.time();assert remaining>600
    p.update(author_stop=deadline,budget_seconds=math.floor(remaining),recovery='Resume ten unfinished videos; native-resolution refinement only for conflicting image correspondences; same window, no object labels or metric outcomes',
        original_worker_sha256=sha(ORIGINAL),original_protocol_sha256=sha(OLD/'protocol.json'),completed_before_resume=done['videos'])
    p['sha256']=dict(p['sha256'])
    for name in ['scripts/b123_supported_transfer_recovery2_20261027.py','tracking/video_alignment_native_recovery.py']:
        p['sha256'][name]=sha(ROOT/name)
    RUN.mkdir();write(RUN/'protocol.json',p);copied={}
    for arm in ['baseline','supported']:
        (RUN/arm).mkdir()
        for v in done['videos']:
            source=OLD/arm/(v+'.json');dest=RUN/arm/(v+'.json');shutil.copy2(source,dest);assert sha(source)==sha(dest);copied[source.relative_to(ROOT).as_posix()]=sha(source)
        shutil.copy2(OLD/arm/'per_video.json',RUN/arm/'per_video.json')
    for v in done['videos']:
        shutil.copytree(OLD/'cache'/v,RUN/'cache'/v)
        for source in (OLD/'cache'/v).rglob('*'):
            if source.is_file():assert sha(source)==sha(RUN/'cache'/v/source.relative_to(OLD/'cache'/v));copied[source.relative_to(ROOT).as_posix()]=sha(source)
    write(RUN/'completed_videos.json',done)
    from tracking.video_alignment_native_recovery import build_alignment
    start=time.monotonic();preflight={}
    for v in p['videos'][10:]:
        assert time.monotonic()-start<120 and time.time()<datetime.fromisoformat(deadline).timestamp()-600
        write(RUN/'status.json',dict(state='preparing',stage='Проверка привязок оставшихся видео без разметки объектов',video=v,updated=datetime.now(timezone.utc).isoformat()))
        images=[ROOT/x['path'] for x in p['images'][v]];output=RUN/'preflight'/v/'alignment.json'
        rows,metadata=build_alignment(ROOT/f'data/bdd100k/videos/val/{v}.mov',images,output,p['search_radius'])
        preflight[v]=dict(sha256=sha(output),refinements=read(output)['native_refinements'],reference_count=len(rows))
    write(RUN/'service_recovery.json',dict(old_controller=end,old_deadline=deadline,scientific_rule_weights_thresholds_unchanged=True,
        completed_videos_copied=done['videos'],copied_source_sha256=copied,preflight=preflight,preflight_seconds=time.monotonic()-start,
        quality_outcomes_used_for_recovery_decision=False,reference_object_labels_used=False))
    from scripts.b123_safe_journal_20261027 import journal
    journal(p['id'],'привязка уточнена по исходным пикселям,10готовыхвидеосохранены',
        '*Ошибка.* После10готовыхвидео уменьшение160×90сопоставило первые два изображения b23b2649-6af03cd5с одним кадром1,строгая проверка остановила опыт. '
        '*Проверено.* Исходные1280×720,640×360и320×180даютодинаковыеуникальныеминимумы0и1;окнопоиска24нерасширялось,разметкаобъектов/частичныеHOTAдлярешениянеиспользовались. '
        'Источник:recovery1/alignment_resolution_diagnostic.json иrecovery2/service_recovery.json. Всеоставшиесяпривязкипредварительнопроверены. '
        '*Возобновление.*10полныхвидео,кэши,выходы,метрикисвереныпохешамископированы;ихYOLO/трекер/оценкаповторятьсянебудут. '
        'Продолжитьтежеоставшиеся10,беззаменывыборки/порогов/правила,старыйдедлайн '+deadline+' сохраняется. Это исправление данныхвремен,неположительныйилиотрицательныйрезультаткачества.')
    print(json.dumps(dict(prepared=True,copied_videos=10,remaining_videos=10,preflight_seconds=time.monotonic()-start,deadline=deadline),ensure_ascii=False))
def cached_alignment(video,images,output,search_radius):
    p=read(RUN/'protocol.json');v=Path(video).stem;service=read(RUN/'service_recovery.json');source=RUN/'preflight'/v/'alignment.json'
    assert search_radius==p['search_radius'] and sha(source)==service['preflight'][v]['sha256']
    obj=read(source);write(output,obj);return obj['matches'],obj['video']
def run():
    p=read(RUN/'protocol.json');assert sha(ORIGINAL)==p['original_worker_sha256'];service=read(RUN/'service_recovery.json')
    for name,h in service['copied_source_sha256'].items():assert sha(ROOT/name)==h,name
    text=ORIGINAL.read_text(encoding='utf-8')
    old="per={arm:{} for arm in ['baseline','supported']};timing={};proof={}"
    new="previous=read(RUN/'completed_videos.json');per={arm:read(RUN/arm/'per_video.json') for arm in ['baseline','supported']};timing=previous['timing'];proof=previous['proof']"
    assert text.count(old)==1;text=text.replace(old,new)
    old="for index,v in enumerate(p['videos']):\n        began=";assert text.count(old)==1;text=text.replace(old,"for index,v in enumerate(p['videos']):\n        if v in timing:continue\n        began=")
    old='from scripts.run_bdd_video_rate_experiment import build_alignment,build_detection_cache,load_detection_cache,detections_for_frame,scalabel_frame,CLASS_NAMES';assert text.count(old)==1
    text=text.replace(old,old+'\n    from scripts.b123_supported_transfer_recovery2_20261027 import cached_alignment as build_alignment')
    old='if index==0:\n            cost=2*timing[v]*19+120';assert text.count(old)==1
    text=text.replace(old,"if index==len(p['completed_before_resume']):\n            cost=2*timing[v]*(len(p['videos'])-index-1)+120")
    write(RUN/'source_transform.json',dict(original_sha256=p['original_worker_sha256'],transformed_sha256=hashlib.sha256(text.encode()).hexdigest(),
        changes=['Load verified completed metrics/cache and skip completed ten videos','Use preflight image-only alignment, same search window','Price only remaining unfinished videos'],scientific_rule_changed=False))
    ns={'__name__':'frozen_worker_import','__file__':str(ORIGINAL)};exec(compile(text,str(ORIGINAL),'exec'),ns);ns['RUN']=RUN;ns['write']=write;ns['run']()
if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    else:
        try:run()
        except Exception as e:write(RUN/'status.json',dict(state='failed',stage=repr(e),updated=datetime.now(timezone.utc).isoformat()));raise
