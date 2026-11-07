"""Отдельное исполнение Б2 после полного аудита весов; только общий срок автора."""
from pathlib import Path
import os,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261028 import RUN,service

NAMES={'gpu':'run_b2_full_tracker_20261029.py','audit':'audit_b2_full_tracker_20261029.py',
       'record':'record_b2_full_tracker_20261029.py','prepare':'prepare_b2_full_tracker_20261029.py'}
WRAPPERS={kind:'scripts/b2_tracker_recovery4_'+kind+'.py' for kind in NAMES}


def code_for(kind):
    source=ROOT/'scripts'/NAMES[kind]
    code=source.read_text(encoding='utf-8')
    code=code.replace('b2_sparse_training_audit','b2_training_gpu_predictor_audit')
    code=code.replace('b2_tracker_preparation','b2_tracker_preparation_recovery4')
    if kind=='prepare':
        for k in ['gpu','audit','record']:
            code=code.replace('scripts/'+NAMES[k],WRAPPERS[k])
        anchor="    protocol=dict(id='b2_predetection_tracker_sparse_v1'"
        extra="    files += "+repr(['scripts/'+NAMES[k] for k in ['gpu','audit','record']]+['tracking/b2_cpu_loader_recovery3.py','scripts/b2_tracker_recovery4_20261029.py'])+"\n"
        assert code.count(anchor)==1;code=code.replace(anchor,extra+anchor)
        code=code.replace('budget_seconds=13000','budget_seconds=86400')
        code=code.replace("gpu_absolute_end='2026-10-29T04:48:00+03:00',audit_absolute_end='2026-10-29T07:18:00+03:00'",
            "gpu_absolute_end=service.read(RUN/'authorization.json')['save_at'],audit_absolute_end=service.read(RUN/'authorization.json')['save_at']")
        code=code.replace("[('b2_tracker_audit',9000),('b2_tracker_record',600)]","[('b2_tracker_audit',86400),('b2_tracker_record',86400)]")
        assert code.count('fits=save-time.time()>13000+9000+600+120')==1
        code=code.replace('fits=save-time.time()>13000+9000+600+120','fits=save-time.time()>300  # Global save reservation only; stage budgets cancelled')
        code=code.replace('full_chain_fits_before_save=fits','minimum_preparation_time_available=fits,full_chain_fits_before_save=None,stage_time_budgets_cancelled=True')
        anchor="    service.write(folder/'protocol.json',protocol)"
        assert code.count(anchor)==1
        code=code.replace(anchor,"    protocol['price']='Time forecasts are advisory after direct author decision; first complete video is measured; disk gate remains mandatory. Global save07:30/stop07:40 only, no promise of complete GPU+CPU series.'\n"+anchor)
        code=code.replace("('130','b2_tracker_pilot'","('146','b2_tracker_pilot'")
        code=code.replace("('135','b2_tracker_audit'","('149','b2_tracker_audit'").replace("('138','b2_tracker_record'","('152','b2_tracker_record'")
        # Прогноз времени сохраняется, дисковая защита остаётся обязательной.
        code=code.replace('False meanssavefirst/remaining,norepeat/noextension/noscientificnegative.',
            'Stage time forecast advisory after direct author cancellation; disk gate remains binding; no scientific negative from incompleteness.')
        code=code.replace('Mainhypothesisgraph11texts/colorsprotected.','B1-B3texts/colorsandnewtrackingbranchprotected.')
        code=code.replace("'Новые13000GPU/9000CPU/600record,первыйполныйвидеопроход/диск/цена,никакогопродленияматрицыистории. '",
            "'Меньшиевременныебюджетыотмененыавтором,общийsave07:30/stop07:40. Первыйполныйвидеопрофиль/дисковаязащитаобязательны. '")
        code=code.replace("    service.write(folder/'protocol.json',protocol)","    protocol['training_audit_device']='cuda, original predictor, separate process; CPU dev-loss equivalence failed and is not claimed'\n    protocol['failed_cpu_training_audit_sha256']=service.sha(RUN/'b2_sparse_training_audit_recovery3/controller_result.json')\n    service.write(folder/'protocol.json',protocol)")
    elif kind=='gpu':
        old="price=dict(proceed=required<available and disk_need<free,first_full_video_seconds=item['seconds'],"
        assert code.count(old)==1
        code=code.replace(old,"price=dict(proceed=disk_need<free,time_forecast_fits=required<available,stage_time_gate_cancelled=True,first_full_video_seconds=item['seconds'],")
    elif kind=='audit':
        old="models=load_branches(torch.load(ROOT/freeze['checkpoint'],map_location='cpu',weights_only=False),'cpu')"
        assert code.count(old)==1
        code=code.replace(old,"models=load_cpu_branches(torch.load(ROOT/freeze['checkpoint'],map_location='cpu',weights_only=False))")
        assert code.count('if need>=available:')==1
        code=code.replace('if need>=available:','if False:  # Stage time gate cancelled by author')
    else:
        code=code.replace('меньшие бюджеты не продлевать и готовое не повторять.','общий срок07:40сохранён,готовоене повторять.')
    return code,source


def main(kind):
    from scripts.history_stage_budget_policy_20261029 import policy
    policy()
    folder=RUN/('b2_tracker_preparation_recovery4' if kind=='prepare' else 'b2_tracker_pilot')
    p=service.read(folder/'protocol.json')
    code,source=code_for(kind)
    assert service.sha(source)==p['sha256'][source.relative_to(ROOT).as_posix()]
    if kind=='prepare':
        for name in ['history_tracker_author_record1','b2_training_gpu_predictor_audit']:
            c=service.read(RUN/name/'controller_result.json')
            assert c['exit_code']==0 and c['all_owned_ended'] and not any(service.matching(k,v) for k,v in c['owned'].items())
        assert service.read(RUN/'history_tracker_author_audit1/report.json')['passed']
        a=service.read(RUN/'b2_training_gpu_predictor_audit/report.json')
        assert a['passed'] and a['complete'] and a['losses_checked']==1200
        assert service.sha(RUN/'b2_sparse_training_audit/controller_result.json')==p['original_failed_audit_sha256']
        service.write(folder/'dependency_verified.json',dict(training_audit_sha256=service.sha(RUN/'b2_training_gpu_predictor_audit/report.json'),
            original_failed_audit_sha256=p['original_failed_audit_sha256'],updated=service.now()))
    ns={'__name__':'b2_tracker_recovery4_'+kind,'__file__':str(source)}
    if kind=='audit':
        from tracking.b2_cpu_loader_recovery3 import load_cpu_branches
        ns['load_cpu_branches']=load_cpu_branches
    exec(compile(code,str(source),'exec'),ns);ns['main']()


if __name__=='__main__':raise RuntimeError('Use dedicated entry point')
