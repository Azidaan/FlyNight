"""Заморозить готовые2эпоховые веса и одинаковые40историческихval доCUDA."""
from pathlib import Path
from datetime import datetime
import sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261028 import RUN,service
read,write,sha,now=service.read,service.write,service.sha,service.now


def main():
    trained=RUN/'history_two_epoch_pilot';audit=RUN/'history_two_epoch_audit';folder=RUN/'history_tracker_pilot';output=RUN/'history_tracker_preparation'
    for f in [trained,audit]:
        c=read(f/'controller_result.json');assert c['exit_code']==0 and c['all_owned_ended'] and not any(service.matching(k,v) for k,v in c['owned'].items())
    assert read(audit/'report.json')['passed'] and read(audit/'report.json')['full_training']
    r=read(trained/'report.json');assert r['complete'] and r['models']==30
    budget=10800;cpu_audit=7200;record=900;checks=600
    remaining=datetime.fromisoformat(read(RUN/'authorization.json')['save_at']).timestamp()-time.time()
    if remaining<budget+cpu_audit+record+checks+180:
        write(output/'report.json',dict(prepared=False,scientific_verdict=None,reason='Full registered chain does not fit before save; no val launched',remaining_seconds=remaining,updated=now()));return
    assert not (folder/'protocol.json').exists();tp=read(trained/'protocol.json')
    models={}
    for m in r['completed']:
        checkpoint=trained/'models'/f"{m['variant']}_seed{m['seed']}"/'best.pt'
        assert sha(checkpoint)==m['best_sha256'] and len(m['epochs'])==2
        models[f"v{m['variant']}_seed{m['seed']}"]=dict(variant=m['variant'],seed=m['seed'],checkpoint=checkpoint.relative_to(ROOT).as_posix(),sha256=m['best_sha256'])
    frozen=dict(created=now(),models=models,alpha=.2,epochs=2,selected_without_val=True,training_protocol_sha256=sha(trained/'protocol.json'),training_audit_sha256=sha(audit/'report.json'))
    write(folder/'frozen.json',frozen)
    old=read(ROOT/'runs/recovery_research_20261027/lazy_appearance_validation/protocol.json')
    sources=[{k:s[k] for k in ['video','group','baseline','baseline_metrics','detections','alignment','video_path','raw_array','raw_array_key']} for s in old['sources']]
    config=read(RUN/'history_pair_dataset/protocol.json')['tracker_config'];assert config==old['config']
    files=['scripts/run_history_full_tracker_pilot_20261028.py','scripts/audit_history_full_tracker_20261028.py','scripts/summarize_history_tracker_20261028.py',
        'tracking/history_pair_features.py','tracking/history_pair_model.py','tracking/history_pair_model_deterministic.py','tracking/history_pair_runtime.py','tracking/history_training_data.py',
        'tracking/causal_motion_history.py','tracking/masa_r50.py','tracking/frame_masa_appearance.py','tracking/xywh_bytetrack.py','weights/masa/masa_r50.pth',
        'runs/z13_val20_matched_20261024/native_masa_recovery2/report.json','runs/continuation_20261028/history_tracker_pilot/frozen.json']
    hashes={f:sha(ROOT/f) for f in files}
    for s in sources:
        for k in ['baseline','detections','alignment','video_path','raw_array']:
            path=s[k];assert sha(ROOT/path)==old['sha256'][path];hashes[path]=old['sha256'][path]
    visual=read(RUN/'history_visual_inputs/protocol.json')
    p=dict(created=now(),parent_experiment='history_factorial_tracker_two_epoch_v1',budget_seconds=budget,sources=sources,groups=old['groups'],
        tracker_config=config,alpha=.2,variants=tp['variants'],seeds=tp['seeds'],epochs=2,motion=visual['motion'],context=visual['context'],image_size=visual['image_size'],
        input_coverage='Всеqueryдетекциисscore>=.1,независимоотназначения/GT. История каждогоиз32трекеровраздельная,только реальноназначенныеqueryобласти. ОбщийкэшсохраняеткаждыйдоступныйROIдляCPUаудита,необмениваетсяID.',
        capacity_limit='45537номинальныхпараметроввсем. ВсеGRUвычисляют6кандидатов,но активныеобновлениясостоянияуисториизависятотмаски,уlast/meanагрегатповторяется6раз. Это сравнениецелыхпредставлений/реализаций,нестрогоеотделениеодноготолькопорядкаприравномчислеактивныхобновлений. Унулевогоканаладавностинетинформационнойнагрузки.',
        action=tp['future_action'],labels=tp['labels'],head_cpu_gpu_tolerance=tp['audit_forward_tolerance'],
        validation='40старыхисторическихval;не новыйнезависимыйитоговыйтест. Alpha/головы/сочетания замороженыДОCUDA,не выбираютсяпоval. 2эпоховыйпилотневыдаётсязавыполнение6эпох.',
        price_rule='ПервыйПОЛНЫЙвидеопроход32трекеров/входов/метрик:2*time*39+900<остаток10800с;иначестопбезпродленияссохранениемпервого. Готовыйпервыйнеповторять.',
        audit='CPUнезависимоповторяетвходы/головус3e-5/геометрию/память/оригинальныйcost. Для побитного повтора дискретных назначений послеэтойпроверкииспользуетзаписанныйGPUscore;неутверждатьпобитностьCPU/GPUголовы. Всеraw/JSON/нулевыесостояния/метрики/парныеинтервалыточны;пикселивидео0/20/39повторены.',
        primary='134vsnative/0/mean/1meanвкаждомиз3сидов;все12положительныхнижнихграницпослеBonferroni;обычные95%ДИтакжепубликуются. Остальныесочетанияипарные/тройноевзаимодействияописательные. Эффектынескладывать.',sha256=hashes)
    write(folder/'protocol.json',p)
    history=read(ROOT/'experiments/project_history.json');ident='history_factorial_tracker_two_epoch_v1';assert not any(c['id']==ident for c in history['experiments'])
    history['experiments'].append(dict(id=ident,title='Полный трекер с тремя представлениями истории — пилот двух эпох',area='flyvis',state='ready',started='2026-10-28',
        question='Даётлисовместнаяистория1/3/4добавочнуюHOTAкбазе/последнему/среднемуприодинаковом2эпоховомобучении?',
        hypothesis='Предварительнопроверяемаягипотезапользы:134превосходитсогласованныепростыеконтроливовсех3начальныхзначениях;2эпохиограничиваютвывод.',
        expected='Все32полныхтрекера(native/zero/30голов)наодних40историческихval,полныеHOTA/AssA/IDF1/FP/FN/IDSw,парныеДИ/12первичныхсравнений/интервалывзаимодействий. Ниодногоподборапоval.',
        acceptance=[dict(criterion=x,met=None) for x in ['Полные40видео/32трекера,независимыйаудитназначений/головы/сырыхвыходов/метрик;нулевойконтрольточеннакаждомкадре',
            '134положителенкnative/0/mean/1meanвовсех3сидахпонижнейскорректированнойграницепарнойHOTA(12сравнений)',
            'Веса/alpha/источникизамороженыдопрогона,GTнеподаётсявпредиктор,всеисториистрогопрошлые,6эпоховыйценовойстопсохранён']],
        evidence=[dict(kind='file',ref=folder.relative_to(ROOT).as_posix()+'/protocol.json',claim='Научныеусловия/источники/цена/аудитдоCUDA'),dict(kind='file',ref=folder.relative_to(ROOT).as_posix()+'/frozen.json',claim='Готовыевеса30голов/alphaдоval')],metrics=[],confirmed_results=[]))
    write(ROOT/'experiments/project_history.json',history)
    for name,sub,s,b in [('085_history_tracker_checks','history_tracker_checks','scripts/check_history_tracker_pilot_20261028.py',checks),
        ('090_history_tracker_validation','history_tracker_pilot','scripts/run_history_full_tracker_pilot_20261028.py',budget),
        ('095_history_tracker_audit','history_tracker_audit','scripts/audit_history_full_tracker_20261028.py',cpu_audit),
        ('098_history_tracker_record','history_tracker_record','scripts/record_history_full_tracker_20261028.py',record)]:
        f=RUN/sub
        if sub!='history_tracker_pilot':write(f/'protocol.json',dict(parent_experiment=ident,budget_seconds=b,sha256={s:sha(ROOT/s)}))
        write(RUN/'ready_jobs'/(name+'.json'),dict(name=name,folder=f.relative_to(ROOT).as_posix(),script=s,script_sha256=sha(ROOT/s),protocol_sha256=sha(f/'protocol.json')))
    write(output/'report.json',dict(prepared=True,frozen_sha256=sha(folder/'frozen.json'),models=30,trackers=32,epochs=2,videos=40,updated=now()))
    from scripts.b123_safe_journal_20261027 import journal
    journal(ident,'Зарегистрирован полный32вариантныйтрекерныйпилотисториина40видео',
        'Источник history_tracker_pilot/protocol.json/frozen.json:доCUDAзафиксированы30двухэпоховыхголов/alpha0,2/40историческихval. '
        'ОбщийпричинныйизвлекательсохраняетВСЕqueryROIпорогу0,1,память/IDраздельны;GTнетвпредикторе. '
        '085обязательныепроверки→090GPU10800сспервымПОЛНЫМ32трекернымвидеогейтом→095CPU7200саудит→098запись900с. '
        '12первичныхсравнений/Bonferroni/3сида;сыройzero/stateточныnative;CPUголовапроверяетсяспрежней3e-5,дляточныхдискретныхназначенийиспользуетсяпроверенныйзаписанныйGPUscore. '
        'Результат2эпохневыдаётсязавыполнениестарых6эпох/независимыйновыйтест/вкладядраFlyVis.')
    with (RUN/'HANDOFF.md').open('a',encoding='utf-8') as f:
        f.write('\n\nПосле082passedготоваПОЛНАЯцепочка085CPUchecks600→090history_tracker_pilotGPU10800→095history_tracker_auditCPU7200→098record900. '
            'Frozen30весов/alpha.2/2эпохиДОCUDA;теже40историческихval,не20и не новыйитоговыйтест. '
            'ОдинобщийизвлекателькодируетВСЕqueryROIscore>=.1,неbaselineunion. Все32живыхтрекерараздельны/native/zero/30голов. '
            'ПервоеПОЛНОЕвидео2*time*39+900<остаток10800,приfalseсохранитьпервый/цену/остатокбезпродления/ненаучныйnegative. '
            'GTнетвпредикторе. АудитCPUвновьвычисляетголову/геометрию/входы;допускголовы3e-5ПРЕЖНИЙ. ПослеэтойпроверкизаписанныйGPUscoreиспользуетсядляпобитного повторадискретныхназначений;невыдаватьCPU/GPUголовыпобитными. '
            'Всеraw/JSON/state/метрики/парныеинтервалыповторяются;пиксели0/20/39;checkpointпослеполноговидео. '
            'Главноесравнение134сnative/0/mean/1meanв3сидах/12первичныхCIпослеBonferroni. Полныеметрикиисточникивreport.json/independent_audit.json. '
            'После098нуженАГЕНТдлянезависимогопрочтенияитога/частныхподтверждений/проблем/фундаментальногообзора. Не приписывать ядруFlyVis;оно здесьнеиспользуется.\n')


if __name__=='__main__':main()
