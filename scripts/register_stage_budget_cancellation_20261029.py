"""Регистрирует прямую отмену меньших бюджетов, сохраняя общий срок и историю."""
from pathlib import Path
import json
import sys
from datetime import datetime, timezone, timedelta
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261028 import RUN,service


def main():
    folder=RUN/'stage_budget_policy_20261029';folder.mkdir(exist_ok=True)
    assert service.matching(9664,1793203064.7696502)
    assert service.matching(20528,1793203064.802624)
    assert service.read(RUN/'queue_status.json')['job']=='120_history_author_gpu.json'
    auth=service.read(RUN/'authorization.json')
    service.write(folder/'authorization.json',dict(
        human_message='Только бюджеты этапов; общую остановку в 07:40 оставить',
        human_response_source='Direct send_user_message_question_reply in this chat',
        question_item_id='["request_user_input_async","call_98ecaa63ea4142ebb10efb9b3e6ed126",0]',
        stage_time_limits_cancelled=True,hard_stop=auth['hard_stop'],save_at=auth['save_at'],
        controller_stop=auth['controller_stop'],created=service.now(),
        scientific_protocol_unchanged=True,completed_work_repeated=False))
    from scripts.history_stage_budget_policy_20261029 import transformed
    src=RUN/'history_tracker_author_continuation1';aud=RUN/'history_tracker_author_audit1'
    checks={}
    for kind in ['gpu','audit','record']:
        code,origin=transformed(kind,src,aud);compile(code,str(origin),'exec')
        checks[kind]=dict(compiled=True,original_sha256=service.sha(origin))
    for name in ['125_history_author_audit.json','128_history_author_record.json']:
        path=RUN/'ready_jobs'/name
        assert not (RUN/'jobs_done'/name).exists()
        old=service.read(path);prev=folder/'previous_pending_jobs'/name
        assert not prev.exists();service.write(prev,old)
        protocol=ROOT/old['folder']/'protocol.json'
        service.write(folder/'previous_protocols'/name,service.read(protocol))
    def publish(name,stage,script):
        stage.mkdir(parents=True,exist_ok=True)
        protocol=dict(budget_seconds=86400,
            budget_note='Compatibility value only; controller is bounded by unchanged global save_at',
            stage_time_limits_cancelled=True,policy_sha256=service.sha(folder/'authorization.json'),
            parent_experiment='history_factorial_tracker_two_epoch_v1',
            sha256={script:service.sha(ROOT/script),
                    'scripts/history_stage_budget_policy_20261029.py':service.sha(ROOT/'scripts/history_stage_budget_policy_20261029.py')})
        service.write(stage/'protocol.json',protocol)
        service.write(RUN/'ready_jobs'/(name+'.json'),dict(name=name,folder=stage.relative_to(ROOT).as_posix(),
            script=script,script_sha256=service.sha(ROOT/script),protocol_sha256=service.sha(stage/'protocol.json')))
    # 121 не повторяет GPU после успеха120; при его временном стопе принимает только законченные видео поSHA.
    publish('121_history_stage_time_followup',folder/'gpu_followup','scripts/history_stage_time_gpu.py')
    publish('125_history_author_audit',aud,'scripts/history_stage_time_audit.py')
    publish('128_history_author_record',RUN/'history_tracker_author_record1','scripts/history_stage_time_record.py')
    proof=dict(checks=checks,live_gpu_unchanged=True,
        live_worker_captured_old_deadline=True,
        conditional_followup='121 only if old120 hits its cancelled time limit; completed videos never repeated',
        guard_deadline_unchanged=service.read(RUN/'deadline_status.json')['stop_at'],updated=service.now())
    service.write(folder/'registration_verified.json',proof)
    ms=datetime.now(timezone(timedelta(hours=3))).strftime('%d.%m.%Y, %H:%M')
    text=('\n\nПРИОРИТЕТНОЕ РЕШЕНИЕ АВТОРА '+ms+' МСК: «Только бюджеты этапов; общую остановку в 07:40 оставить». '
          'AGENTS9.1 уточнён: меньшие бюджеты/временные costgate отменены как самостоятельная причина остановки; '
          'общие save07:30/guard07:39:30/stop07:40 сохранены. История старых стопов не переписана. '
          'RUN/stage_budget_policy_20261029/authorization.json содержит точный человеческий ответ. '
          '120 уже загрузила старый срок в память; её не прерывать ради перенастройки. Новая121 в той же очереди '
          'после успешной120 только фиксирует выбор результата без GPU; если120 остановится по прежнему времени, '
          'продолжает только незавершённые видео в отдельной папке, готовые пропускает поSHA. '
          'При tracking_completed безreport требуется толькоCPUвосстановление метрик, GPUповтор запрещён. '
          '125/128 ещё не запускались, старые задания/протоколы сохранены в previous_pending_jobs/previous_protocols; '
          'новые обёртки ограничены общим save07:30,125 не останавливается по прогнозу длительности первогоCPUвидео. '
          'Выбор фактического GPUисточника в stage_budget_policy_20261029/selected_gpu_result.json. '
          'Все научные проверки/источники/модели/допуски сохранены; реквизиты в registration_verified.json. '
          'Б2остаётся отложенной по приоритету матрицы; отмена времени не является подтверждением её аудита.\n')
    with (RUN/'HANDOFF.md').open('a',encoding='utf-8') as f:f.write(text)
    p=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';old=p.read_text(encoding='utf-8')
    assert old.count('## Журнал изменений\n')==1
    entry='\n\n**'+ms+' MSK** — Автор отменил меньшие бюджеты этапов\n\n'+text.strip()+'\n'
    p.write_text(old.replace('## Журнал изменений\n','## Журнал изменений\n'+entry,1),encoding='utf-8')
    print(json.dumps(proof,ensure_ascii=False))


if __name__=='__main__':main()
