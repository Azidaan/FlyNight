"""В существующую очередь после результата матрицы; новый GPU не запускается."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261028 import RUN,service


def main():
    from scripts.history_stage_budget_policy_20261029 import policy
    policy()
    proof=RUN/'b2_cpu_loader_recovery3/report.json';r=service.read(proof)
    assert r['passed'] and r['no_cuda_initialized'] and not r['training_repeated']
    launch=service.read(proof.parent/'launch.json');assert not service.matching(launch['pid'],launch['created'])
    assert service.matching(9664,1793203064.7696502) and service.matching(20528,1793203064.802624)
    folder=RUN/'b2_sparse_training_audit_recovery3'
    assert not folder.exists()
    old=RUN/'b2_sparse_training_audit';p=service.read(old/'protocol.json')
    p.update(budget_seconds=86400,
        budget_note='Stage time budgets cancelled by author; controller remains bounded by global save07:30',
        original_failure_sha256=service.sha(old/'controller_result.json'),cpu_loading_proof_sha256=service.sha(proof),
        created=service.now(),scope='Same strict1200CPUlosschecks;explicitCPUdevices;noCUDA;notHOTA')
    files=['scripts/audit_b2_sparse_training_recovery3_20261029.py','scripts/audit_b2_sparse_training_recovery2_20261029.py',
        'tracking/b2_cpu_loader_recovery3.py']
    p['sha256'].update({s:service.sha(ROOT/s) for s in files})
    service.write(folder/'protocol.json',p)
    script=files[0];name='129_b2_sparse_training_audit_recovery3'
    service.write(RUN/'ready_jobs'/(name+'.json'),dict(name=name,folder=folder.relative_to(ROOT).as_posix(),script=script,
        script_sha256=service.sha(ROOT/script),protocol_sha256=service.sha(folder/'protocol.json')))
    now=datetime.now(timezone(timedelta(hours=3))).strftime('%d.%m.%Y, %H:%M')
    text=('**'+now+' MSK** — Проверено исправление CPU-загрузки Б2; строгий аудит поставлен после матрицы\n\n'
        'Исходный116bexit1/deviceCUDAсохранён. Перваяотдельнаяпроверка b2_cpu_loader_recovery1 отказала: '
        'flyvis.network.initialization сохранил CUDAdevice через from-import и создавал несовместимый генератор; '
        'вторая отказала на неоднозначном импорте flyvis.network, обеошибки/исходники/PIDendсохранены отдельно. '
        'Третья tracking/b2_cpu_loader_recovery3.py явно задаётCPUвflyvis,initialization,decoder через importlib; '
        'официальнаябиблиотека/окружение/весане изменены. b2_cpu_loader_recovery3/report.json passed: '
        'все6ветвей/всепараметрыпобитносохранённомуcheckpoint,CUDAnotinitialized,обучениенеповторялось. '
        'Это проверка загрузки, НЕполныйаудит/НЕdevloss/НЕHOTA. '
        '129_b2_sparse_training_audit_recovery3 уже опубликована в существующей очереди ПОСЛЕ120/121/125/128. '
        'Старыеаудитор/контроллер/папканепереписываются;новаяверсиясохраняет все1200CPUlossсравнений/3e-5/источники/optimizer/corechecks/видеочекпойнты. '
        'Меньшийвременнойcostgateотменёнпоавторскомурешению,общийsave07:30/stop07:40неизменён. '
        'После129НУЖЕНАГЕНТ:полныйreport/controller/PIDend. ТолькоpassedcompleteаудитдопускаетподготовкуБ2fulltracker. '
        'Старыйprepare_b2_full_tracker требуетложногоуспеха116b/старых1200с/13000+9000fit; '
        'нельзя запускать его какесть: нужнаотдельнаяверсиясзависимостьюуспешной129+SHAстарогоотказа, '
        'CPUloaderrecovery3вполномаудите,заморозкойдоval/новымфактическимпрофилем/глобальнымсроком. '
        'Готоваяматрица/B1/Б2обучениенеповторяются.\n\n')
    plan=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';s=plan.read_text(encoding='utf-8');assert s.count('## Журнал изменений\n')==1
    plan.write_text(s.replace('## Журнал изменений\n','## Журнал изменений\n\n'+text,1),encoding='utf-8')
    with (RUN/'HANDOFF.md').open('a',encoding='utf-8') as f:f.write('\n\n'+text)
    service.write(RUN/'b2_cpu_recovery3_registered.json',dict(job=name,loading_report_sha256=service.sha(proof),
        original_failed_controller_sha256=service.sha(old/'controller_result.json'),loading_pid_ended=True,
        source_sha256={s:service.sha(ROOT/s) for s in files},updated=service.now()))
    print(json.dumps(dict(queued=name,scope='CPU only after history result',loading_exact=True,training_repeated=False),ensure_ascii=False))


if __name__=='__main__':main()
