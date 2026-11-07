"""Зарегистрировать новое окно и досчёт только одного видео Б1."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import sys,os,shutil,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261028 import RUN,service
read,write,sha,now=service.read,service.write,service.sha,service.now
from scripts.b123_safe_journal_20261027 import journal
ARMS=['baseline','appearance','zero','C','D','R1','B']

def main():
    assert not (RUN/'authorization.json').exists(),'Не повторять подготовленное окно'
    oldroot=ROOT/'runs/recovery_research_20261027';old=oldroot/'b1_tracker/continuation_recovery1'
    for path in [oldroot/'queue_recovery1/launch.json',oldroot/'guard.launch.json',old/'launch.json',oldroot/'b1_partial_audit/launch.json']:
        d=read(path);assert not service.matching(d['pid'],d['created']),path
    for name in ['ready_jobs','jobs_done','b1_tracker','b1_tracker_audit','preflight']:(RUN/name).mkdir(parents=True,exist_ok=True)
    write(RUN/'authorization.json',dict(created=now(),user_request='продолжи',basis='Новое прямое разрешение после возвращения автора; AGENTS.md9.1 до следующего буднего07:40МСК',
        hard_stop='2026-10-29T07:40:00+03:00',save_at='2026-10-29T07:30:00+03:00',controller_stop='2026-10-29T07:39:30+03:00',
        scope='Досчитать последнее видеоБ1, полный независимый аудит с сохранением по видео; затем подготовка и проверка представлений1/3/4 отдельно и совместно, Б2/Б3 отложены.',single_gpu=True,
        limits=['Новых данных/весов/смены окружения нет','Старые бюджеты/ошибки не изменять','Готовые19GPUвидео не повторять','Основнойbaseline/11гипотез/цвета не менять','.git/index.lock не трогать']))
    write(RUN/'registry.json',dict(processes=[]))
    oldp=read(old/'protocol.json');videos=[s['video'] for s in oldp['sources'] if (old/'completed'/(s['video']+'.json')).exists()];assert len(videos)==19
    manifest=read(oldroot/'b1_partial_audit/partial_source.json')['sha256']
    for path,digest in manifest.items():assert sha(ROOT/path)==digest,path
    seed=dict(completed_videos=videos,per_video=read(old/'per_video.json'),proof={},timing={},output_sha256={},
        old_audit_sha256=sha(oldroot/'b1_tracker/recovery2/independent_audit.json'),source_manifest_sha256=sha(oldroot/'b1_partial_audit/partial_source.json'))
    folder=RUN/'b1_tracker'
    for v in videos:
        marker=read(old/'completed'/(v+'.json'));seed['proof'][v]=marker['proof'];seed['timing'][v]=marker['seconds']
        write(folder/'completed'/(v+'.json'),dict(marker,reused_without_inference=True,source=(old/'completed'/(v+'.json')).relative_to(ROOT).as_posix()))
        for arm in ARMS:
            for sub,ext in [(arm,'.json'),('events/'+arm,'.json'),('raw/'+arm,'.npz')]:
                source=old/sub/(v+ext);target=folder/sub/(v+ext);target.parent.mkdir(parents=True,exist_ok=True)
                assert not target.exists();os.link(source,target)
                expected=sha(source);assert sha(target)==expected
                seed['output_sha256'][target.relative_to(ROOT).as_posix()]=expected
    for arm in ARMS:assert set(seed['per_video'][arm])==set(videos)
    write(folder/'resume_seed.json',seed);write(folder/'per_video.json',seed['per_video'])
    shutil.copy2(old/'geometry.npz',folder/'geometry_seed.npz')
    from scripts.run_b1_last_video_20261028 import worker_source
    from scripts.audit_b1_checkpointed_20261028 import audit_source
    worker='scripts/run_b1_last_video_20261028.py';auditor='scripts/audit_b1_checkpointed_20261028.py'
    scientific=dict(config=oldp['config'],sources=oldp['sources'],strong_reference=oldp['strong_reference'],limitation=oldp['limitation'])
    hashes=dict(oldp['sha256']);hashes.update({p:sha(ROOT/p) for p in [worker,auditor,'scripts/continuation_service_20261028.py','tracking/b1_audit_checkpoints.py',
        'tracking/b1_cached_eye.py','tracking/b1_deterministic_runtime.py', (folder/'resume_seed.json').relative_to(ROOT).as_posix()]})
    write(folder/'protocol.json',dict(**scientific,parent_experiment='b1_full_tracker_pilot_v1',created=now(),budget_seconds=3600,
        sha256=hashes,executed_worker_sha256=hashlib.sha256(worker_source().encode()).hexdigest(),
        original_protocol=(old/'protocol.json').relative_to(ROOT).as_posix(),original_protocol_sha256=sha(old/'protocol.json'),
        same_scientific_settings=True,continuation_scope='Reuse19bySHA before decode; infer only last. Same7variants/weights/seed/threshold/precision.',
        new_user_permission='продолжи',original_cost_gate_not_rewritten=True))
    write(RUN/'b1_tracker_audit/protocol.json',dict(parent_experiment='b1_full_tracker_pilot_v1',budget_seconds=7200,
        sha256={p:sha(ROOT/p) for p in [auditor,'scripts/audit_b1_tracker_20261028.py','tracking/b1_audit_checkpoints.py']},
        executed_audit_sha256=hashlib.sha256(audit_source().encode()).hexdigest(),
        checkpoint_policy='Exact independent audit saved per whole video with inputSHA. First complete old audit reused after matching all inputs. Interrupted uncheckpointed parts are not treated as completed.'))
    write(RUN/'preflight/protocol.json',dict(budget_seconds=600,parent_experiment='b1_full_tracker_pilot_v1',sha256={p:sha(ROOT/p) for p in ['scripts/preflight_b1_resumption_20261028.py','tests/test_b1_audit_checkpoints.py']}))
    history=read(ROOT/'experiments/project_history.json');card=next(c for c in history['experiments'] if c['id']=='b1_full_tracker_pilot_v1')
    fields=dict(state='running',done='',decision='По новому прямому поручению28.10«продолжи» досчитать только последнее видеоБ1. Готовые19переиспользуютсяпоSHA безинференса. Затем полный20аудит с сохранением по видео; прежние сроки и отказы не переписываются. Новое окно до29.10 07:40МСК. После результата—представления1/3/4, без повторения готового.')
    for key,value in fields.items():
        card.setdefault('revisions',[]).append(dict(date='2026-10-28',commit='',field=key,old=card.get(key,''),reason='Новое прямое поручение продолжить в следующем разрешённом окне'))
        card[key]=value
    card['evidence'].append(dict(kind='file',ref=(RUN/'authorization.json').relative_to(ROOT).as_posix(),claim='Новое разрешение; готовые19не пересчитываются'))
    for a in card['acceptance']:a['met']=None;a['evidence']='Новая попытка завершения той же серии; приёмка только после20и полного аудита'
    write(ROOT/'experiments/project_history.json',history)
    write(RUN/'b1_active_stage.json',dict(source='b1_tracker',audit='b1_tracker_audit',updated=now()))
    for name,sub,script in [('000_preflight','preflight','scripts/preflight_b1_resumption_20261028.py'),('010_b1_last_video','b1_tracker',worker),('020_b1_full_audit','b1_tracker_audit',auditor)]:
        f=RUN/sub;write(RUN/'ready_jobs'/(name+'.json'),dict(name=name,folder=f.relative_to(ROOT).as_posix(),script=script,script_sha256=sha(ROOT/script),protocol_sha256=sha(f/'protocol.json')))
    write(RUN/'preparation.json',dict(reused_videos=videos,remaining_videos=[s['video'] for s in oldp['sources'] if s['video'] not in videos],
        old_sources_unchanged=True,new_outputs_linked_immutable=True,sha256=seed['output_sha256'],updated=now()))
    (RUN/'HANDOFF.md').write_text('# Продолжение28.10.2026\n\nНовое прямое поручение «продолжи»,окно до29.10 07:40МСК,save07:30/kill07:39:30. '
        '000проверки→010только последнееБ1видео(3600с)→020полныйаудит20(7200с)ссохранениемпо видео. Готовые19используютсяпоSHA,нетпереобучения/сменывесов/порогов. '
        'Первое ранее завершённое независимоеCPUподтверждение переиспользуетсяпосверке;прежний19аудитне завершён и неможетсчитатьсяготовым. '
        'Старыеисточники/протоколы/контроллеры/ошибки сохранены вruns/recovery_research_20261027. Новаяqueue/guard толькоэтогоокна,неподниматьстарые. '
        'После020нуженагент: полный report/independent_audit/controllers/PID+birth, научныекритерииDкbase/appearance/C/R1/B,старыйMASAнаодних20,карточка/revisions/журнал/обязательныепроверки/фундаментальныйграф. '
        'Затемреальныевходы1/3/4/общийобучаемыйинтерфейс/стоимость/полнадаментальнаяHOTA;Б2/Б3отложены. Матрица0,1,3,4,13,14,34,134ипростыепоследний/EMAконтроли. '
        'ОдинGPU/скрытые процессы/нетновыхданных,весов,среды/baseline/.git/index.lockне менять/11гипотезпобайтно.\n',encoding='utf-8')
    journal('b1_resumption_20261028','Новое окно:досчёт последнего видеоБ1и полный аудит с сохранением по видео',
        '*Постановка.* По прямому«продолжи»автора `runs/continuation_20261028/authorization.json`: до29.10 07:40МСК. '
        '*Получено.* `preparation.json`:готовые19свереныпоSHAи используютсябезинференса;осталосьодно видео. '
        '*Решение.*000проверкикаталога/тесты→010GPU3600с→020CPU7200с,вновомокне,непродлеваястарые. Нуженполныйаудит20ссохранениемпо видео. Веса/пороги/7вариантовнаучноне меняются. ПослеБ1—представления1/3/4. Результатовновойсериипока нет.')
    print(RUN,'prepared')

if __name__=='__main__':main()
