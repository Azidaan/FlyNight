"""Проверить принятое решение человека, фактический запуск и точный остаток Б2."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261028 import RUN,service


def main():
    folder=RUN/'history_tracker_author_continuation1'
    auth=service.read(folder/'author_override.json')
    assert auth['accepted'] and auth['human_message']=='Продолжить матрицу до утра'
    l=service.read(folder/'launch.json');assert service.matching(l['pid'],l['created'])
    c=service.read(RUN/'history_author_preparation1/controller_result.json')
    assert c['exit_code']==0 and c['all_owned_ended'] and not any(service.matching(k,v) for k,v in c['owned'].items())
    seed=service.read(folder/'resume_seed.json')
    assert seed['first_gpu_video_repeated'] is False
    assert all(service.sha(ROOT/f)==h for f,h in seed['sha256'].items())
    original=RUN/'history_tracker_pilot'
    assert service.sha(original/'controller_result.json')==seed['source_controller_sha256']
    assert service.sha(original/'report.json')==seed['source_report_sha256']
    assert service.read(original/'price.json')['proceed'] is False
    guard=service.read(RUN/'deadline_status.json')
    assert abs(guard['owned'][str(l['pid'])]-l['created'])<.01
    assert service.matching(9664,1793203064.7696502) and service.matching(20528,1793203064.802624)
    pending=[x.name for x in (RUN/'ready_jobs').glob('*.json') if not(RUN/'jobs_done'/x.name).exists()]
    assert set(pending).issubset({'120_history_author_gpu.json','125_history_author_audit.json','128_history_author_record.json'})
    b2=service.read(RUN/'b2_sparse_training_pilot/report.json')
    assert b2['complete'] and all(x==840 for x in b2['optimizer_steps'].values())
    b2c=service.read(RUN/'b2_sparse_training_audit/controller_result.json')
    assert b2c['exit_code']==1 and b2c['all_owned_ended'] and not any(service.matching(k,v) for k,v in b2c['owned'].items())
    proof=dict(updated=service.now(),human_authorization_sha256=service.sha(folder/'author_override.json'),
        gpu_launch=l,status=service.read(folder/'status.json'),execution_deadline=service.read(folder/'execution_deadline.json'),
        first_gpu_reused_sha_files=len(seed['sha256']),old_cost_stop_unchanged=True,guard_updated=guard['updated'],pending=pending,
        b2_training_complete=True,b2_training_report_sha256=service.sha(RUN/'b2_sparse_training_pilot/report.json'),
        b2_training_final_sha256=b2['final']['sha256'],b2_cpu_audit_complete=False,
        b2_audit_failed_controller_sha256=service.sha(RUN/'b2_sparse_training_audit/controller_result.json'),
        b2_audit_error_sha256=service.sha(RUN/'b2_sparse_training_audit/stderr.log'),
        b2_audit_old_deadline=service.read(RUN/'b2_sparse_training_audit/execution_deadline.json')['deadline'],
        b2_full_val_gpu_queued=False)
    service.write(RUN/'history_human_continuation_verified.json',proof)
    text='''

ПРИОРИТЕТНОЕПРЯМОЕРЕШЕНИЕАВТОРА29.10около00:33МСК: на прежний асинхронный вопрос получен ЧЕЛОВЕЧЕСКИЙответ «Продолжить матрицу до утра». Это новый прямой ответ send_user_message_question_reply, НЕheartbeat/accepted:true. ВСЕПРЕЖНИЕзаписи«ответанет/39недопущены»этимрешениемЗАМЕНЕНЫвчастипродолжения;историческиестопы/ценыНЕпереписаны. Точныйответ/вопрос/idсохраненывhistory_tracker_author_continuation1/author_override.json:accepted=true,scope=Continue same frozen32tracker matrix beyond original090cost gate with new explicit stage deadlines,gpu_end29.10 04:00МСК,audit_end07:20МСК. Общиеsave07:30/stopowned07:39:30/07:40неизменны. Повторноспрашиватьэтотже выборНЕнужно.

117_history_author_preparation600сCPUПОЛНОСТЬЮготова/code0/all_owned_ended/реальныедеревьяended/catalogOK/60passed127.01с. СЕЙЧАС120_history_author_gpuРЕАЛЬНОвыполняетсявhistory_tracker_author_continuation1. GPUlaunchPID25008+birth1793223393.6806457,execution_deadline2026-10-29T00:59:15.677025UTC=03:59:15МСК(меньше04:00;НЕпродлевать). ЭтоОДИНновыйпрогонТОЛЬКОоставшихся39,теже30готовых2эпоховыхголов/3сида/.2/32живыхтрекера/40историческихval/12Bonferroniсравнений. Первоеb1d10d08-743fd86cподключенопоSHA/hardlinks,report/resume_seedподготовленыДОдекода,workerперепроверилSHAипропустилЕГОдоинференса. НЕповторятьпервыйGPU/обучение/старыйценовойгейт. Новыйprice.jsonсохраняетстарыйfalseиauthor_override:true;отрицательныйстарыйproceedНЕповодостановкиновогоразрешения. Первыйновыйреальныйвидеопроходindex1 b23b2649-8349d2a1уже дошёлдоHOTAна00:38МСК;этоснимок,читатьсвежийstatus/report/launch/controller.

Следующие125_history_author_audit и128_history_author_recordУЖЕопубликованы,не создаватьдубли. 125folderhistory_tracker_author_audit1,budget11955с/абсолютныйcap07:20,реальныйexecution_deadlineпослестартаГЛАВНЕЕиНЕпродлевается. ПолныйCPUповтор40/32,головавпрежнем3e-5,посленезависимогопересчётазаписанныйGPUscoreдляпобитногоповторадискретныхназначений. ИсправленТОЛЬКОисходныйNumPy2типcostчерезhistory_cost_replay.py,20891событие exact;старыйаудитнепереписан. Метрики/paired/zero/raw/JSON/mean/covточны;пиксели0/20/39/checkpointпослецелоговидеосSHA. ПервыйПОЛНЫЙCPUвидеогейтпрежний,еслиfalseнеобходить/непродлеватьегоавтоматически. 128folderhistory_tracker_author_record1,budget540с,старыйrecordsourceсновымипутями/cardrevisions/частнымиconfirmed_results/каталог/граф. ПОСЛЕ120/125/128НУЖЕНАГЕНТ:прочитатьполныеreport/independent_audit/контроллеры/фактическийконецВСЕХPID+birth,40ключей/SHAпервогорeused,12критериев/частныеэффекты/ограничения,обновитьфундаментальныйобзор/Проблемы. ПолнаяHOTA40ПОКАНЕГОТОВА;невыдаватьпрогрессзаитог.

Б2ПОНОВОМУЧЕЛОВЕЧЕСКОМУПРИОРИТЕТУОТЛОЖЕНА. 116обучениеПОЛНОСТЬЮготово/code0/allended:6ветвейC/D/R1/B/last/mean,840optimizerstepsкаждой,200devцелейкаждой,1эпохаseed26/10целейнавидео,финальныйcheckpointсoptimizer/SHAсохранён;неповторятьобучение. 116arecordготова/code0/allended/catalogOK/62passed138.04с. Источникb2_sparse_training_pilot/report.json. Но116bCPUauditУПАЛАДОпересчётаdevloss:официальныйFlyVisget_from_state_dictзагрузилstockчерезmap_location=flyvis.device,выбранCUDA0приCPUэтапе. Исходныйcontrollerexit1/allended/stderr/input_manifestсохранены,НЕпереписыватьуспехом. Старый1200сdeadline116b=2026-10-28T21:50:47.260002UTC(00:50:47МСК),НЕпродлевать. Старый118_b2_tracker_preparationexit1назависимости116b,готовойБ2GPUвалидацииНЕТ. Возможноеисправление—передbuild_armявноactivate_flyvis();flyvis.device=torch.device('cpu');torch.set_default_device('cpu')исверитьреальнуюCPUзагрузку. ЭТОПОКАПЛАН,НЕвалидированноеисправление;библиотеку/окружение/весаНЕменяли. ПолногонезависимогоаудитаБ2/devloss/quality/HOTAНЕТ;техническийсбойзагрузчикаНЕотрицательнаянаука.

РеализациябудущегополногоБ2valнаписана,НОНЕПУБЛИКОВАНА/НЕЗАПУЩЕНА:tracking/b2_pilot_runtime.py,run_b2_full_tracker_20261029.py,audit_b2_full_tracker_20261029.py,summarize_b2_tracker_20261029.py,record_b2_full_tracker_20261029.py,prepare_b2_full_tracker_20261029.py. 6CPUunitвместепрошли5.13с(4входа+1изолированныйtracker50кадров+1статистикапоказывающаячтоD-nativeнедостаточно). НикакогоБ2fullvalпротокола/карточкиb2_predetection_tracker_sparse_v1/runtimeгейтаещёНЕТ:118отказал ДОних. ЧерновиккодаБ2планировал13000GPU/9000CPU/40/8/полныеrawhead+дисковыйгейт,НОЭТОНЕзарегистрированные/НЕразрешённыеактивныеБ2бюджеты. ПослеавторскогоприоритетаматрицыНЕпубликоватьБ2цепочкуавтоматически,НЕпереноситьновыйoverrideматрицынаБ2. Сначалаполныйитогматрицы/остаток/условиястарогобюджета;при необходимостиНОВОЕпрямоерешениечеловека. ГотовыеБ2веса сохраняются,fullHOTAБ2неизмерена.

history_human_continuation_verified.jsonсодержитпроверкуновогоPID+birth/registry/свежегоguard,первогоreusedSHA/старогостопа/всехготовыхБ2деревьев. PendingТОЛЬКО120/125/128,дублей/GPUБ2нет. Однаqueue9664+birth1793203064.7696502иguard20528+birth1793203064.802624,какпрежде.11гипотез/граф/старыеP1/index.lockсохранены;commit/pushнебыли. ВовремядлительныхcatalogtestsНЕПИСАТЬплан/каталог/словарь/контролируемыеreports. При неизменномходенеуведомлять. НикакихновыхMASAветвей. СтароеБ1готово,неповторять.
'''
    with (RUN/'HANDOFF.md').open('a',encoding='utf-8') as out:out.write(text)
    from scripts.b123_safe_journal_20261027 import journal
    journal('history_author_continuation1','Новое решение автора исполнено: матрица продолжена с сохранённого первого видео',
        'Точныйчеловеческийответ «Продолжить матрицу до утра» в history_tracker_author_continuation1/author_override.json. '
        '117code0/allended/catalogOK/60passed127,01с;120GPUреальносчитаетindex1,первоеGPUпоSHAпропущенододекода. '
        'НепродлеваемыйновыйGPUexecution_deadline29.10 03:59:15МСК,аудитcap07:20,общий07:40неизменён. '
        'Б2обучение6ветвей/840updates/200devкаждойготово;CPUаудит116bотказалназагрузкеmap_locationCUDAприCPU, '
        'отказисходныйсохранён/непольза/ненегативнаянаука;полныйБ2valне запускалсяиотложенподновыйприоритет. '
        'Источникпроверки всехPID+birth/SHA:history_human_continuation_verified.json. '
        'Ранее два infrastructure recorderБ2не сохранили прежний cannot_claim вrevisions:точные тексты сохраненывисходныхпротоколах/скриптах; '
        'нужновосстановитькопиивrevisionsприследующейбезопаснойправкекаталога,последующейвалидатор/pytest;нево времяживогоcatalogsuite.')


if __name__=='__main__':main()
