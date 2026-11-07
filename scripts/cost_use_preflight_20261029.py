"""Read-only wrapper checks and registration of already authorized cost chain."""
from pathlib import Path
import sys,inspect
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261029 import RUN,service

def main():
    from scripts import cost_use_dev_20261029 as dev,cost_use_val_20261029 as val
    d=inspect.getsource(dev.engine.main);v=inspect.getsource(val.engine.main)
    assert d.count("trackers.update({n+'_guarded':AssignmentWriteTracker(args,s,p['appearance_weight']) for n,s in servers.items()})")==1
    assert d.count("for n in ['r50_guarded','detic_guarded']:")==1
    assert v.count("comparisons={e+'_'+b:compare(per[e+'_cautious'],per[e+'_'+b]) for e in ['r50','detic'] for b in ['last','mean']}")==1
    assert v.count("native_comparisons={n:compare(per[n],per['native']) for n in names}")==1
    assert v.count('primary_comparisons=comparisons,native_comparisons=native_comparisons,pooled=pooled,')==1
    for n in ['queue','guard']:
        launch=service.read(RUN/(n+'.launch.json'));assert service.matching(launch['pid'],launch['created'])
    from datetime import datetime,timezone
    status=service.read(RUN/'deadline_status.json')
    assert (datetime.now(timezone.utc)-datetime.fromisoformat(status['updated'])).total_seconds()<60
    assert datetime.now(timezone.utc)<datetime.fromisoformat(service.read(RUN/'authorization.json')['save_at'])
    jobs=[('300','cost_use_preparation','scripts/prepare_cost_use_dev_20261029.py'),('330','cost_use_val_preparation','scripts/prepare_cost_use_val_20261029.py')]
    for index,n,s in jobs:
        assert not (RUN/n/'protocol.json').exists()
        service.write(RUN/n/'protocol.json',dict(budget_seconds=86400,created=service.now(),scope='Preparation gate only; dependencies and all catalog checks must pass before new scientific jobs are published',authorization_sha256=service.sha(RUN/'cost_use_authorization.json')))
        service.write(RUN/f'ready_jobs/{index}_{n}.json',dict(name=n,folder=str((RUN/n).relative_to(ROOT)),script=s,script_sha256=service.sha(ROOT/s),protocol_sha256=service.sha(RUN/n/'protocol.json')))
    (RUN/'HANDOFF.md').open('a',encoding='utf-8').write('\nПРИОРИТЕТ: автор ПРЯМО ответил «Да, проверить применение признаков» на вопрос call_2f59e972e4b2496abe707cbabc77f86e; cost_use_authorization.json/decision_pending acceptedtrue. Старое ожидание ответа закрыто. В той же очереди300cost_use_preparation→310devCPU→320dev_auditCPU→330val_preparation→340valCPU→350val_auditCPU→360record. 310/320 публикует300толькопослекаталогатестов;340/350/360публикует330толькопослеполногоdev/аудита/фактическогоPIDend/каталогатестов. НЕДУБЛИРОВАТЬ. Последняяпамять/YOLO/query/геометрия/вес0,1прежние: old c+.1d / centered c+.1(d-.5) / общийshift-only c-.05. Триновыхr50_centered/detic_centered/r50_shift,точныестарыеGPUFP32векторыпоSHAбезencoder/GPU/декода/обучения. 4primarycentered-old/shift×2encoder/20kseed1/95%иBonf98,75%,mainвсе4lower>0. Триnativeвторичные,частныеconfirmed_resultsтолькопридополнительнойfamily7ci_global7lower>0,дописанододевпробыбезсменыглавногокритерия. Извлекателиописательно. ПриcatalogtestsНЕПИСАТЬплан/каталог/словарь/контролируемыеreports. Всеошибкисохранять,готовое20/старыеопытынеповторять. После360АГЕНТполныеотчёты/4primary/3native/pooled/описательноеизвлекателей/все20SHA/фактическийPIDend/карточка/подтверждения/граф. Общийsave30.10 07:30/ownedstop07:39:30/hardstop07:40неизменен.\n')
    print('Existing queue and guard verified; authorized cost chain registered.')

if __name__=='__main__':main()
