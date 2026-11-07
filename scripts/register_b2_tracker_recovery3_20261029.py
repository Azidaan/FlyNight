"""Подготовка будущей Б2 после аудита; сейчас GPU не запускает и каталог не меняет."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261028 import RUN,service


def main():
    from scripts.history_stage_budget_policy_20261029 import policy
    policy()
    assert service.matching(9664,1793203064.7696502) and service.matching(20528,1793203064.802624)
    from scripts.b2_tracker_recovery3_20261029 import code_for,WRAPPERS,NAMES
    compiled={}
    for kind in NAMES:
        code,original=code_for(kind);compile(code,str(original),'exec')
        compiled[kind]=service.sha(original)
    folder=RUN/'b2_tracker_preparation_recovery3';assert not folder.exists()
    files=['scripts/'+NAMES['prepare'],'scripts/b2_tracker_recovery3_20261029.py',*WRAPPERS.values(),
        'tracking/b2_cpu_loader_recovery3.py','tests/test_b2_tracker_recovery3_policy.py']
    p=dict(budget_seconds=86400,parent='b2_predetection_tracker_sparse_v1',created=service.now(),
        note='Conditional CPUpreparation only: requires complete passed129 and finished historyrecord. NoGPUbeforetests/freeze.',
        original_failed_audit_sha256=service.sha(RUN/'b2_sparse_training_audit/controller_result.json'),
        sha256={f:service.sha(ROOT/f) for f in files})
    service.write(folder/'protocol.json',p)
    name='130_b2_tracker_preparation_recovery3';script=WRAPPERS['prepare']
    service.write(RUN/'ready_jobs'/(name+'.json'),dict(name=name,folder=folder.relative_to(ROOT).as_posix(),script=script,
        script_sha256=service.sha(ROOT/script),protocol_sha256=service.sha(folder/'protocol.json')))
    source=RUN/'history_tracker_author_continuation1';r=service.read(source/'report.json');c=service.read(source/'controller_result.json')
    assert r['complete'] and len(r['completed'])==40 and len({e['video'] for e in r['completed']})==40
    assert len(r['per_video'])==32 and all(len(v)==40 for v in r['per_video'].values())
    assert c['exit_code']==0 and c['all_owned_ended'] and not any(service.matching(k,v) for k,v in c['owned'].items())
    seed=service.read(source/'resume_seed.json');assert seed['first_gpu_video_repeated'] is False
    assert all(service.sha(ROOT/f)==h for f,h in seed['sha256'].items())
    first=r['completed'][0];assert first['video']==seed['video'] and first['reused_before_decode']
    assert first['sha256']==seed['sha256']
    audit=RUN/'history_tracker_author_audit1';launch=service.read(audit/'launch.json')
    assert service.matching(launch['pid'],launch['created'])
    service.write(RUN/'history_gpu40_completion_verified.json',dict(complete=True,videos=40,trackers=32,
        all_owned_ended=True,controller_sha256=service.sha(source/'controller_result.json'),
        report_sha256=service.sha(source/'report.json'),first_gpu_video_repeated=False,first_reused_sha_verified=True,
        scientific_verdict_published=False,independent_audit_running=True,audit_launch=launch,updated=service.now()))
    now=datetime.now(timezone(timedelta(hours=3))).strftime('%d.%m.%Y, %H:%M')
    text=('**'+now+' MSK** — Матрица40GPUполностью сохранена, аудит идёт; условная подготовка полногоБ2готова\n\n'
        '120code0/all_owned_ended/фактическивсеPID+birthended;report.complete=true/40уникальныхвидео/32варианта, '
        'первыйreusedGPU/всеSHAперепроверены,неповторён. history_gpu40_completion_verified.json. '
        '125CPUаудитреальновыполняется;прогнозпервогополного73,766с,6053,748с<17226,915с. '
        'Сырыеитоговыестатистикипрочитаны,НОдоpassedполногоаудитанаучныйвердиктНЕпубликуется. '
        '129CPUаудитБ2следуетпосле128. Новая130_b2_tracker_preparation_recovery3УЖЕвсуществующейочереди, '
        'проверяетполныеpassed129/1200loss+конецдеревьев/успешный128/аудитматрицы;раньшеэтогоGPUнепубликуется. '
        'Отдельный scripts/b2_tracker_recovery3_20261029.py клонируетстарыйподготовленныйпутьБ2, '
        'неизменныепервые40/native/zero/C/D/R1/B/last/mean/1эпохаseed26/исходныечисленныедопуски. '
        'Старые116b/118отказысохранены,зависимостьтеперьна129+SHAстарогоотказа;CPUполныйаудитиспользуетloaderrecovery3. '
        'Новое130послепроверокзамораживаетпротокол/весаДОval,карточку/каталог/64tests,затемготовит131GPU/135CPU/138record. '
        'Временныеforecastгейтыинепродлеваемыеменьшиеcapзамененыавторскимобщимsave07:30/stop07:40; '
        'дисковыйгейтпервогоПОЛНОГОвидеоосталсяОБЯЗАТЕЛЬНЫМ. Фактическуюценусохранять. '
        'full_chain_fits_before_save=None:вместимостьвсейцепочкиНЕобещается. '
        '8CPUtests5,16спроверилиотменувременногогейтабезотменыдисковогои сохранностьassertпроверок. '
        'СейчаснетновойHOTAБ2/новыхобучений/готовогоБ2GPU;подготовкатолькоусловная. '
        'При130catalogtestsнеизменятьплан/каталог/словарь/контролируемыеисточники. '
        'Еслиtracking_completedБ2есть,аметрикинет—восстанавливатьТОЛЬКОCPU,неповторятьGPU.\n\n')
    plan=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';s=plan.read_text(encoding='utf-8');assert s.count('## Журнал изменений\n')==1
    plan.write_text(s.replace('## Журнал изменений\n','## Журнал изменений\n\n'+text,1),encoding='utf-8')
    with (RUN/'HANDOFF.md').open('a',encoding='utf-8') as f:f.write('\n\n'+text)
    service.write(RUN/'b2_tracker_recovery3_prepared.json',dict(job=name,source_hashes=p['sha256'],
        compiled_originals=compiled,unit_tests=dict(passed=8,seconds=5.16),conditional_only=True,updated=service.now()))
    print(json.dumps(dict(conditional_job=name,history_GPU40_verified=True,new_GPU_started=False),ensure_ascii=False))


if __name__=='__main__':main()
