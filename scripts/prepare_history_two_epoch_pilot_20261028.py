"""Новая дешёвая научная версия по цене, без ослабления старой приёмки6эпох."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
from datetime import datetime
import time,sys,shutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261028 import RUN,service


def main():
    original=RUN/'history_factorial_training';previous=RUN/'history_factorial_training_recovery1';audit=RUN/'history_factorial_training_audit_recovery1'
    for f in [previous,audit]:
        c=service.read(f/'controller_result.json');assert c['exit_code']==0 and c['all_owned_ended'] and not any(service.matching(k,v) for k,v in c['owned'].items())
    r=service.read(previous/'report.json');assert not r['complete'] and r['stopped_on_cost'] and r['saved_epoch']==1
    old_price=service.read(previous/'price.json');assert not old_price['proceed']
    folder=RUN/'history_two_epoch_pilot';assert not (folder/'protocol.json').exists()
    end=datetime.fromisoformat(service.read(original/'execution_deadline.json')['deadline']).timestamp()
    budget=int(end-time.time()-990);need=2*old_price['first_full_epoch_seconds']*59+300
    decision=dict(epochs=2,models=30,first_epoch_reused=True,remaining_epochs=59,first_full_epoch_seconds=old_price['first_full_epoch_seconds'],conservative_seconds=need,available_seconds=budget-20,proceed=need<budget-20,
        changed_from_six_epoch_series=True,selection_basis='Elapsed compute time only; no val or quality selection',original_absolute_deadline=datetime.fromtimestamp(end).astimezone().isoformat())
    service.write(folder/'price.json',decision)
    if not decision['proceed']:
        service.write(folder/'admission.json',dict(admitted=False,scientific_verdict=None,reason='Even separately scoped two-epoch pilot does not fit original window',decision=decision,updated=service.now()));return
    import torch
    assert not torch.cuda.is_initialized()
    prior=previous/'models/0_seed42';resume=torch.load(prior/'resume.pt',map_location='cpu',weights_only=False);best=torch.load(prior/'best.pt',map_location='cpu',weights_only=False)
    assert len(resume['epochs'])==1 and best['epoch']==1 and best['variant']=='0' and best['seed']==42
    target=folder/'models/0_seed42';target.mkdir(parents=True)
    copied={}
    for src in prior.rglob('*'):
        if not src.is_file():continue
        rel=src.relative_to(prior);dst=target/rel;dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,dst);assert service.sha(dst)==service.sha(src);copied[rel.as_posix()]=service.sha(src)
    p=service.read(previous/'protocol.json');p.update(created=service.now(),parent_experiment='history_factorial_two_epoch_pilot_v1',budget_seconds=budget,epochs=2,
        scope='Отдельный ограниченный2эпохами пилот той же архитектуры/матрицы/3начальныхзначений. Исходная6эпоховаясерия остановлена поцене и НЕвыполнена; её критерии/источники не изменяются.',
        selection='Ровно2эпохи;checkpoint минимумdevloss/приравенстверанний. Длина выбрана только по фактической стоимости,безval/выбора качества. Не доказательство достаточности2эпох.',
        reused_epoch=dict(folder=prior.relative_to(ROOT).as_posix(),copied_sha256=copied,best_sha256=service.sha(prior/'best.pt'),protocol_sha256=best['protocol_sha256'],optimizer_reused=True,first_epoch_repeated=False),
        price_rule='Первый полныйtrain+devэпоха0seed42УЖЕсохранён:2*time*59+300<остатокстарого075минус900саудит/90срезерв. Готовая эпоха иoptimizerпереиспользуются,не повторяются. Нового5400снет.')
    for name in ['scripts/train_history_two_epoch_pilot_20261028.py','scripts/audit_history_two_epoch_pilot_20261028.py']:
        p['sha256'][name]=service.sha(ROOT/name)
    service.write(folder/'protocol.json',p)
    history=service.read(ROOT/'experiments/project_history.json');ident='history_factorial_two_epoch_pilot_v1';assert not any(c['id']==ident for c in history['experiments'])
    history['experiments'].append(dict(id=ident,title='Ограниченный двумя эпохами пилот представлений истории',area='flyvis',state='ready',started='2026-10-28',
        question='Можно ли получить сопоставимые веса всехпредставлений/контролейпри2эпохахи3начальныхзначениях,сохранивпервуюготовуюэпоху?',
        hypothesis='Отдельная гипотезаготовностиограниченногопилота:в оставшемся прежнем окне помещаются2эпохи всех30голов;качество поHOTAещёнепроверяется.',
        expected='30голов одной архитектуры/3начальныхзначения/РОВНО2эпохи/fit84fit20/unknownмаска/прежняявзвешеннаяBCE;ниодногоvalдлявыбора. Готоваяперваяэпоха0seed42иoptimizerпереиспользуютсяпоSHA.',
        acceptance=[dict(criterion=s,met=None) for s in ['Полные30голов/2эпохисодинаковойархитектурой/инициализацией/данными;цена новойверсиипрошла,первыйготовыйepochнеповторён',
            'GTтолькоtargets,unknownисключён,весаклассовтолькоfit84;источники/разбиения/хешинеизменны',
            'Независимоповторенывсеdevпотериипервыйполныйdevвидеопроходкаждойголовы;6эпоховаясерияНЕобъявленауспешной']],
        evidence=[dict(kind='file',ref=folder.relative_to(ROOT).as_posix()+'/protocol.json',claim='Новая2эпоховаяверсиядообучения:прежнийконец/источники/копииoptimizer/ограничения'),dict(kind='file',ref=previous.relative_to(ROOT).as_posix()+'/price.json',claim='Исходный6эпоховыйценовойстопсохранён')],metrics=[],confirmed_results=[]))
    service.write(ROOT/'experiments/project_history.json',history)
    # Обязательные проверки каталога перед запуском новой научной версии.
    import subprocess
    for i,args in enumerate([['scripts/validate_project_history.py'],['-m','pytest','tests/test_project_history.py','-q']]):
        with (folder/f'precheck_{i}.log').open('w',encoding='utf-8') as out:result=subprocess.run([sys.executable,*args],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,timeout=300)
        assert result.returncode==0
    # Фактический остаток уменьшился на CPUпроверки. Не продлевать установленную границу.
    actual_budget=int(end-time.time()-990);decision['available_seconds']=actual_budget-20;decision['proceed']=need<actual_budget-20
    service.write(folder/'price.json',decision)
    if not decision['proceed']:
        service.write(folder/'admission.json',dict(admitted=False,scientific_verdict=None,reason='After mandatory checks budget gate no longer passes',decision=decision,updated=service.now()));return
    p['budget_seconds']=actual_budget;service.write(folder/'protocol.json',p)
    s='scripts/train_history_two_epoch_pilot_20261028.py'
    service.write(RUN/'ready_jobs/081_history_two_epoch_pilot.json',dict(name='081_history_two_epoch_pilot',folder=folder.relative_to(ROOT).as_posix(),script=s,script_sha256=service.sha(ROOT/s),protocol_sha256=service.sha(folder/'protocol.json')))
    service.write(folder/'admission.json',dict(admitted=True,scientific_verdict=None,decision=decision,updated=service.now()))
    from scripts.b123_safe_journal_20261027 import journal
    journal(ident,'После ценового стопа6эпох зарегистрирован отдельный2эпоховыйпилот',
        'history_factorial_training_recovery1/price.json:перваяПОЛНАЯэпоха27,375с,прогноз10100,25с>4103,5276с;6эпоховаясерияНЕвыполнена,цена/источникисохранены. '
        'Новаяверсия history_two_epoch_pilot/protocol.json:РОВНО2эпохи всех30голов/3начальныхзначений/прежняяматрица/потери/источники. '
        'Первыйготовыйepoch0seed42иoptimizerскопированыпоSHAбезповтора. Решениеиспользуеттолькостоимость,невиделоval/качествадлявыборадлины. '
        'Новыйпрогноз2*27,375*59+300=3530,25с,новыйGPUбюджет'+str(actual_budget)+'сТОЛЬКОостатокисходного075конца20:14:37UTCминус900аудит/90резерв. '
        'Приёмка6эпоховойсерииНЕослаблена,еёcardinconclusive. Этоотдельныйпилотготовностивесов,неHOTA/доказательстводостаточности2эпох.')
    with (RUN/'HANDOFF.md').open('a',encoding='utf-8') as f:
        f.write('\n\n075/0776эпохСТОППОЦЕНЕ:27.375с/полнуюtrain+devэпоху,10100.25>4103.5276с.077code0/allended,078сохраниланеполноту/карточкуinconclusive/каталог. '
            'ОтдельныйНОВЫЙ2эпоховыйпилот history_factorial_two_epoch_pilot_v1:081GPU'+str(actual_budget)+'сТОЛЬКОостатокстарого075абсолютногоконца20:14:37UTCминус900саудит/90срезерв. '
            'Ниодинстарыйбюджетнепродлён,6эпоховыекритериинеослаблены/неуспешны. Длина2выбранатолькопофактическойценебезval/качества. '
            'Готовыйпервыйepoch0seed42иoptimizerвосстановленыпоSHAсотдельнымикопиями,неповторять. Ровно2эпохи всех30голов/3сида. '
            'Цена3530.25<остатокпроверенапослеобязательныхcatalogtests;price.jsonужеproceed,неждатьповторногопервогогейта. '
            'После081самапубликуется082CPUаудит900свнутритогожеисходного075окна. Путиhistory_two_epoch_pilot/report.jsonиhistory_two_epoch_audit/report.json. '
            'Старые075/077/080/078источникиостаются;075failed/077costfalseнеподменятьуспехом. Этооднаограниченнаянаучнаяверсия,небесконечныйрестарт6эпох. '
            'После082нужнаинтеграция32полныхтрекеров/общиеvalвходы/HOTA/проверки;2эпохи—ограничение,неполноеподтверждениевсейматрицы.\n')


if __name__=='__main__':main()
