"""Точный новый ответ человека; не ответ таймера и не accepted от вопроса."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261028 import RUN,service


def main():
    folder=RUN/'history_tracker_author_continuation1'
    assert not(folder/'author_override.json').exists()
    for name in ('b2_sparse_training_pilot','b2_sparse_training_record','b2_sparse_training_audit','b2_tracker_preparation'):
        c=service.read(RUN/name/'controller_result.json')
        assert c['all_owned_ended'] and not any(service.matching(k,v) for k,v in c['owned'].items())
    assert service.matching(9664,1793203064.7696502) and service.matching(20528,1793203064.802624)
    service.write(folder/'author_override.json',dict(accepted=True,human_message='Продолжить матрицу до утра',
        human_question='Разрешаешь отдельное продолжение этой же серии до общего утреннего срока с резервом на аудит, без повторения первого видео?',
        human_response_source='Direct send_user_message_question_reply from the author in this chat; not a heartbeat or tool accepted flag',
        question_item_id='["request_user_input_async","call_eb96aae2e885479ca26295cc46a027d7",0]',
        scope='Continue same frozen32tracker matrix beyond original090cost gate with new explicit stage deadlines',
        gpu_end='2026-10-29T04:00:00+03:00',audit_end='2026-10-29T07:20:00+03:00',
        global_stop_unchanged='2026-10-29T07:40:00+03:00',created=service.now(),
        b2_status='Allsixtrainingbranchescompleted;GPUended. CPUauditfailedbeforedevreplayondevice mapping;alloldfailurespreserved. B2fulltrackerpreparationfaileddependencybeforeval;noB2GPUvalidationqueued. Deferredforauthorhistorypriority.'))
    prep=RUN/'history_author_preparation1';script='scripts/prepare_history_author_continuation_20261028.py'
    service.write(prep/'protocol.json',dict(budget_seconds=600,created=service.now(),
        authorization_sha256=service.sha(folder/'author_override.json'),sha256={script:service.sha(ROOT/script)}))
    service.write(RUN/'ready_jobs/117_history_author_preparation.json',dict(name='history_author_preparation1',folder=prep.relative_to(ROOT).as_posix(),
        script=script,script_sha256=service.sha(ROOT/script),protocol_sha256=service.sha(prep/'protocol.json')))


if __name__=='__main__':main()
