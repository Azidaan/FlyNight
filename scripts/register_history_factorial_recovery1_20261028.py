"""Замена эквивалентного пула после реального отказа, без продления075окна."""
from pathlib import Path
from datetime import datetime
import time,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261028 import RUN,service


def main():
    original=RUN/'history_factorial_training';folder=RUN/'history_factorial_training_recovery1'
    assert not (folder/'protocol.json').exists()
    profile=RUN/'history_deterministic_pool_profile';c=service.read(profile/'controller_result.json')
    assert c['exit_code']==0 and c['all_owned_ended'] and not any(service.matching(k,v) for k,v in c['owned'].items()) and service.read(profile/'report.json')['passed']
    old=service.read(original/'controller_result.json');assert old['exit_code']==1 and old['all_owned_ended'] and not any(service.matching(k,v) for k,v in old['owned'].items())
    assert not list((original/'models').rglob('*.pt'))
    p=service.read(original/'protocol.json');deadline=service.read(original/'execution_deadline.json')['deadline']
    p.update(created=service.now(),original_absolute_deadline=deadline,budget_seconds=int(datetime.fromisoformat(deadline).timestamp()-time.time()-990),
        recovery='AdaptiveAvgPool8x8→4x4 has no deterministic CUDA backward in this build. AvgPool2d2x2 gives exact CPUforward/allgrads and GPUforward, deterministic GPUbackward twice. No optimizer step/checkpoint/epoch existed before failure. Same45537parameters/data/loss/seed/epochs/criteria; original075/080failures preserved.',
        audit_reserve='900s plus90s reserve INSIDE original075absolute window. Old080 is not resumed. The recovered GPUstage and new CPUaudit must BOTH end before old075deadline.')
    assert p['budget_seconds']>300
    for f in ['tracking/history_pair_model_deterministic.py','scripts/train_history_factorial_recovery1_20261028.py','scripts/audit_history_factorial_training_recovery1_20261028.py','runs/continuation_20261028/history_deterministic_pool_profile/report.json']:
        p['sha256'][f]=service.sha(ROOT/f)
    service.write(folder/'protocol.json',p)
    proof=dict(updated=service.now(),reason=p['recovery'],original_deadline=deadline,recovered_gpu_budget=p['budget_seconds'],reserved_cpu_audit_seconds=900,
        old_controller_sha256=service.sha(original/'controller_result.json'),old_audit_controller_sha256=service.sha(RUN/'history_factorial_training_audit/controller_result.json'),
        old_source_sha256=service.sha(ROOT/'scripts/train_history_factorial_20261028.py'),profile=service.read(profile/'report.json'),original_gpu_all_ended=True)
    service.write(folder/'recovery_proof.json',proof)
    script='scripts/train_history_factorial_recovery1_20261028.py'
    service.write(RUN/'ready_jobs/077_history_factorial_training_recovery1.json',dict(name='077_history_factorial_training_recovery1',folder=folder.relative_to(ROOT).as_posix(),script=script,script_sha256=service.sha(ROOT/script),protocol_sha256=service.sha(folder/'protocol.json')))
    from scripts.b123_safe_journal_20261027 import journal
    journal('history_factorial_training_recovery1','Исправлен недоступный детерминированный градиент пула до первой эпохи',
        '075 реально завершиласькодом1/all_owned_ended:adaptive_avg_pool2d_backward_cuda не поддерживаетдетерминированныйрежим. 080сохранилаотказзависимости. '
        'Ниодного optimizerstep/checkpoint/эпохинебыло. Источники/ошибки/контроллерысохранены. '
        '076проверилаэквивалентнуюзамену8x8→4x4наAvgPool2d2x2:CPUforward/всеградиентыпобитны,GPUforwardпобитен,дваGPUbackwardточны,32реальныепары/45537параметров;history_deterministic_pool_profile/report.json. '
        '077вотдельнойпапкеhistory_factorial_training_recovery1,бюджет='+str(p['budget_seconds'])+'сТОЛЬКОостатокПРЕЖНЕГО075абсолютногоокна '+deadline+'минус900саудит/90срезерв. '
        'Готовые065/070неповторяются,ценаПЕРВОЙполнойэпохиостаётсяобязательной,настройкинеизменны. 078CPUаудитсамапубликуетсяпослеуспешноговыходаGPU;новыйаудитоплаченрезервом075,старый080не возобновляется. Нельзяпродлеватьисходныйсрок.')
    with (RUN/'HANDOFF.md').open('a',encoding='utf-8') as f:
        f.write('\n\nАКТУАЛЬНО:060/065полностьюготовы104/182050ROI/5099pixelreplay,065code0/allended/catalogOK/58passed130.35с.070code0/allended. '
            '075УПАЛАДОпервойэпохи/optimizerstep:deterministic adaptive_avg_pool2d_backward_cuda unsupported.080сохранилаотказзависимости,старыеисточники/отказыНЕпереписаны. '
            '076GPUprofilepassed/code0/allended:заменаAdaptiveAvgPool8→4наобычныйAvgPool2x2точнаCPUforward/всеграды/GPUforward,дваGPUbackwardточны;45537параметров/32реальныепары. '
            '077_history_factorial_training_recovery1УЖЕопубликована,папкаhistory_factorial_training_recovery1. GPUбюджет'+str(p['budget_seconds'])+'с—остатокПРЕЖНЕГО075конца'+deadline+'минус900саудит/90срезерв. '
            'Обевосстановленныеступени077GPU/078CPUдолжныполностьюокончитьсяВНУТРИпрежнего075окна;старый080не возобновляется/непродлевается. '
            'ПерваяПОЛНАЯtrain+devэпоха6*30-1гейтобязателен,еслиfalseнепродлевать. После077самаопубликуется078CPUстрогийпрежнийаудитсновымиадресами. '
            'Старую075не выдаватьзауспешную/неждатьreportвстаройпапке. Не повторятьготовыеБ1/входы/профили. Старыйисходный075конец—главнаяграница,не новый5400.\n')
    print(proof)


if __name__=='__main__':main()
