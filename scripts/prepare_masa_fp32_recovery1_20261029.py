"""Explicit precision correction, separate from completed TF32 execution."""
from pathlib import Path
import sys,subprocess,shutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261029 import RUN,service

def main():
    checks=RUN/'masa_fp32_preparation'
    for name in ['masa_query_pilot','masa_query_audit','masa_feature_numerics']:
        c=service.read(RUN/name/'controller_result.json')
        assert c['all_owned_ended'] and not any(service.matching(k,v) for k,v in c['owned'].items())
    proof=service.read(RUN/'masa_feature_numerics/report.json')
    assert proof['complete'] and len(proof['comparisons'])==3
    assert all(x['gpu']['original_flags']['vs_saved']==0 for x in proof['comparisons'])
    assert all(x['gpu']['tf32_off']['vs_cpu_batched']<proof['original_tolerance'] for x in proof['comparisons'])
    old=RUN/'masa_query_pilot/protocol.json';p=service.read(old)
    source_hash=service.sha(old)
    scripts={
        'gpu':('scripts/run_masa_detic_query_pilot_20261029.py','scripts/run_masa_query_fp32_recovery1_20261029.py'),
        'audit':('scripts/audit_masa_detic_query_pilot_20261029.py','scripts/audit_masa_query_fp32_recovery1_20261029.py'),
        'record':('scripts/record_masa_detic_query_pilot_20261029.py','scripts/record_masa_query_fp32_recovery1_20261029.py')}
    for kind,(source,target) in scripts.items():
        code=(ROOT/source).read_text(encoding='utf-8')
        code=code.replace("'masa_query_pilot'","'masa_query_fp32_recovery1'").replace("'masa_query_audit'","'masa_audit_fp32_recovery1'").replace("'masa_query_record'","'masa_record_fp32_recovery1'")
        # Also correct full evidence paths and journal statements in recorder.
        code=code.replace('continuation_20261029/masa_query_audit/','continuation_20261029/masa_audit_fp32_recovery1/')
        if kind=='gpu':
            anchor='    torch.set_num_threads(2);cv2.setNumThreads(1)'
            assert code.count(anchor)==1
            code=code.replace(anchor,anchor+"\n    torch.backends.cudnn.allow_tf32=False\n    torch.backends.cuda.matmul.allow_tf32=False\n    torch.set_float32_matmul_precision('highest')\n    assert not torch.backends.cudnn.allow_tf32 and not torch.backends.cuda.matmul.allow_tf32")
            anchor='    def status(**kw):\n';assert code.count(anchor)==1
            code=code.replace(anchor,anchor+"        assert shutil.disk_usage(ROOT).free>5*1024**3,'Disk5GBreserve'\n")
            code=code.replace('import sys,time,faulthandler','import sys,time,faulthandler,shutil')
            anchor='            if vi==0:\n';assert code.count(anchor)==1
            code=code.replace(anchor,anchor+"                saved_bytes=sum(f.stat().st_size for f in out.rglob('*') if f.is_file())\n                free=shutil.disk_usage(ROOT).free\n                need=2*saved_bytes*19+5*1024**3\n                service.write(folder/'disk_gate.json',dict(passed=need<free,required_bytes=need,free_bytes=free,updated=service.now()))\n                assert need<free,'Disk gate: preserve first result and stop'\n")
        compile(code,target,'exec');assert not (ROOT/target).exists();(ROOT/target).write_text(code,encoding='utf-8')
    folder=RUN/'masa_query_fp32_recovery1';assert not folder.exists();folder.mkdir()
    p.update(created=service.now(),technical_recovery='Disable TF32 convolution/matmul for original declared FP32 output contract; separate precision-corrected execution, not reuse or overwrite of TF32 outputs',
        old_protocol_sha256=source_hash,old_gpu_preserved='runs/continuation_20261029/masa_query_pilot',
        old_audit_failure_sha256=service.sha(RUN/'masa_query_audit/controller_result.json'),
        precision_proof_sha256=service.sha(RUN/'masa_feature_numerics/report.json'),
        precision=dict(parameters='float32',cudnn_allow_tf32=False,matmul_allow_tf32=False,matmul_precision='highest'),
        reason_for_new_execution='Corrected feature arithmetic changes derived vectors and potentially assignments. Old20outputs remain stored; only this new precision is checked against unchanged CPU1e-5. No weights/training/data/action/threshold tuning.')
    p['sha256'].update({target:service.sha(ROOT/target) for _,target in scripts.values()})
    service.write(folder/'protocol.json',p)
    for name in ['masa_audit_fp32_recovery1','masa_record_fp32_recovery1']:
        service.write(RUN/name/'protocol.json',dict(budget_seconds=86400,parent=p['id'],created=service.now()))
    h=service.read(ROOT/'experiments/project_history.json');c=next(x for x in h['experiments'] if x['id']=='reid_masa_detic_v1')
    c.setdefault('evidence',[]).extend([
        dict(kind='file',ref='runs/continuation_20261029/masa_query_audit/controller_result.json',claim='ПервоначальныйCPUаудитотказалнанормированныхпризнаках;полныйдискретныйповтор20проверенотдельнымиcheckpoint'),
        dict(kind='file',ref='runs/continuation_20261029/masa_feature_numerics/report.json',claim='СогласованноевмешательствосTF32объяснилоразницунаодномфиксированномкадре'),
        dict(kind='file',ref='runs/continuation_20261029/masa_query_fp32_recovery1/protocol.json',claim='ОтдельноезаранеезарегистрированноестрогоеFP32исполнение,исходные20сохранены')])
    c['state']='running'
    service.write(ROOT/'experiments/project_history.json',h)
    from scripts.b123_safe_journal_20261027 import journal
    journal(p['id'],'Отдельное полное FP32исполнение после установленного влиянияTF32',
        'Все20GPUвидео первоначально готовы и сохранены. masa_query_audit: все20события/raw/JSON/метрикиповторены,ночисленныйконтрольпризнаковотказал. '
        'masa_feature_numerics/report.json:нафиксированномкадречестныйGPUповторисходногоTF32точенсохранённымвекторам;CPU/GPUрасхождения3.326e-4/2.893e-4/1.545e-4. '
        'ОтключениеTF32припрочихравныхснизилорасхождениедо5.066e-7/4.843e-7/5.537e-7. Этолокализованнаяпричиначисленногоотказанаэтомкадре,непричинанаучныхпроблемпамятиилиБ2. '
        'Новаяmasa_query_fp32_recovery1/protocol.json:теже20/5/веса/вес.1/истории/допуски,ноявноcudnnTF32=false,matmulTF32=false. '
        'ИсходныйGPUнепереписывается/невыдаётсязанеполный. Новоепересчитываниепризнаковиновыхзависимыхназначенийнеобходимоиз-заизмененияарифметики;стараясериянеподменяетсяеёрезультатом. '
        'ДопускCPU1e-5неослаблен,новыхданных/весов/обучениянет. Срок20:00–21:00остаётсяпрогнозомдофактическогозамеравновойточности.')
    for i,args in enumerate([['scripts/validate_project_history.py'],['-m','pytest','tests/test_project_history.py','-q']]):
        with (checks/f'check_{i}.log').open('w',encoding='utf-8') as out:
            status=subprocess.run([sys.executable,*args],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,timeout=420).returncode
        assert status==0,(i,status)
    service.write(checks/'report.json',dict(passed=True,precision_recovery_registered=True,updated=service.now()))
    for index,name,kind in [('040','masa_query_fp32_recovery1','gpu'),('050','masa_audit_fp32_recovery1','audit'),('055','masa_record_fp32_recovery1','record')]:
        script=scripts[kind][1]
        service.write(RUN/f'ready_jobs/{index}_{name}.json',dict(name=name,folder=(RUN/name).relative_to(ROOT).as_posix(),script=script,
            script_sha256=service.sha(ROOT/script),protocol_sha256=service.sha(RUN/name/'protocol.json')))

if __name__=='__main__':main()
