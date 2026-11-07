"""Записать полныйаудированныйитогилиявнуюнеполноту,неподменяя6эпоховыйстоп."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
from datetime import datetime,timezone,timedelta
import sys,subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261028 import RUN,service
read,write,sha,now=service.read,service.write,service.sha,service.now


def main():
    source=RUN/'history_tracker_pilot';audit=RUN/'history_tracker_audit';folder=RUN/'history_tracker_record'
    for f in [source,audit]:
        c=read(f/'controller_result.json');assert c['exit_code']==0 and c['all_owned_ended'] and not any(service.matching(k,v) for k,v in c['owned'].items())
    r=read(source/'report.json');a=read(audit/'report.json');complete=r['complete'] and a['passed'] and a['full_series']
    history=read(ROOT/'experiments/project_history.json');card=next(c for c in history['experiments'] if c['id']=='history_factorial_tracker_two_epoch_v1')
    old_fields={k:card[k] for k in ['obtained','interpretation','decision','cannot_claim'] if card.get(k)}
    for k,v in old_fields.items():card.setdefault('revisions',[]).append(dict(date=datetime.now(timezone(timedelta(hours=3))).date().isoformat(),commit='',field=k,old=v,reason='Итог полного пилота или явная неполнота; прежний текст сохранён'))
    card['done']=datetime.now(timezone(timedelta(hours=3))).date().isoformat()
    ref='runs/continuation_20261028/history_tracker_pilot/report.json';auditref='runs/continuation_20261028/history_tracker_pilot/independent_audit.json'
    if not complete:
        card.update(state='inconclusive',obtained=ref+' и history_tracker_audit/report.json:полнаянаучнаяприёмкане состоялась;complete='+str(r['complete'])+',audit_passed='+str(a['passed'])+'.',
            interpretation='Цена или незавершённый аудит ограничили вывод;неполнота не является научным отрицанием истории.',decision='Сохранитьполныечасти/хеши/цену/неоконченныйостаток;непродлеватьисходныебюджеты.',cannot_claim='Нетполногоаудированногонаучногоитога;выводобесполезностипредставлений/ядраFlyVisнедопустим.')
        card['acceptance'][0].update(met=False,evidence='runs/continuation_20261028/history_tracker_audit/report.json')
        card['acceptance'][1].update(met=None,evidence='Научный критерий не оценён без завершённой технической проверки')
    else:
        positive=r['main_acceptance'];base=r['pooled']['native']['HOTA'];targets={str(seed):r['pooled'][f'v134_seed{seed}']['HOTA'] for seed in [42,43,44]}
        card.update(state='completed_positive' if positive else 'inconclusive',obtained=ref+':полные40видео/32трекера/3начальныхзначения/2эпохи;nativeHOTA='+str(base)+',134HOTA='+str(targets)+',main_acceptance='+str(positive)+'. Независимыйаудит '+auditref+'.',
            interpretation=('Совместнаяисторияпрошла12первичныхсравненийполнойHOTAсучётоммножественности.' if positive else 'Общийкритерийдополнительнойпользы134надвсемипростымиконтролямив3сидахнепройден. Частныеположительныесравнениясохраняютсяотдельно;эквивалентностьнеустановлена.'),
            decision='Сопоставитьодиночные/парныевариантыисильныепростыеконтроли,проанализироватьузелограниченияпоисточникам;неподбиратьalpha/весапоэтимval. Сохранитьвесьпилотиверсии6эпоховогоценовогостопа.',
            cannot_claim='Только2эпохи/ограничение6наблюдений60кадров/3сида/40историческихval,не6эпоховаясерияине новыйнезависимыйитоговыйтест. НельзяприписыватьпользуядруилимодулюFlyVis:ониздесьнеиспользуются. CPUголованепобитнаGPU:головавдопуске3e-5,дискретноеназначениеповтореноспровереннымзаписаннымGPUscore. КонтрастсофлайнMASAна20неизолируетодинизвлекатель.')
        for i,item in enumerate(card['acceptance']):item.update(met=(positive if i==1 else True),evidence=auditref)
        card['evidence'].append(dict(kind='file',ref=auditref,claim='ПолныйCPUповтор40видео/32трекеров/назначений/сырыхвыходов/метрик'))
        card['metrics']=[dict(key='HOTA',value=base,baseline=base,subset='40историческихval;native',source=ref,locator=['pooled','native','HOTA'])]
        for seed,value in targets.items():card['metrics'].append(dict(key='HOTA',value=value,baseline=base,subset='40историческихval;134;начальноезначение'+seed+';2эпохи',source=ref,locator=['pooled','v134_seed'+seed,'HOTA']))
        card['confirmed_results']=[]
        for name,result in r['primary'].items():
            if result['ci95'][0]<=0:continue
            card['confirmed_results'].append(dict(id='history_two_epoch_'+name,title='Совместная история — частное сравнение '+name,category='tracker',metric='HOTA',unit='points',comparison=name,
                subset='40историческихval',conditions='ТежеYOLO/детекции/трекер;2эпохиодинаковогообучения;веса/alphaзамороженыдоval;полныйаудит.',
                limitations='Частное95%сравнение;семейнаяпоправка12сравнений='+str(result['adjusted_ci'][0]>0)+',общийкритерий='+str(positive)+'. Историческийval/2эпохи;не вкладFlyVis.',
                evidence=dict(source=ref,sha256=sha(source/'report.json'),locator=['primary',name])))
        write(folder/'analysis.json',dict(main_acceptance=positive,primary=r['primary'],pooled_target_hota=targets,native_hota=base,relative_pooled_percent={s:100*(v/base-1) for s,v in targets.items()},measured_ceiling_fraction=None,reason='No matched achievable maximum measured;oracle reserves are not ceilings',updated=now()))
    write(ROOT/'experiments/project_history.json',history)
    from scripts.b123_safe_journal_20261027 import journal
    journal('history_tracker_two_epoch_record','Полный трекерныйпилоттрёхпредставленийистории:зафиксированфактическийитог',
        card['obtained']+' '+card['interpretation']+' '+card['cannot_claim']+' Частныеподтверждённыесравнения:confirmed_resultsисходнойкарточкиhistory_factorial_tracker_two_epoch_v1;исходныеотказы/6эпоховыйценовойстопсохранены.')
    checks=[['scripts/validate_project_history.py'],['-m','pytest','tests/test_project_history.py','-q'],['scripts/restore_obsidian_sections_20261027.py'],['-m','pytest','tests/test_obsidian_fundamental_overview.py','-q']]
    for i,args in enumerate(checks):
        with (folder/f'check_{i}.log').open('w',encoding='utf-8') as out:code=subprocess.run([sys.executable,*args],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,timeout=350).returncode
        assert code==0
    write(folder/'report.json',dict(complete=True,scientific_series_complete=bool(complete),scientific_acceptance=r.get('main_acceptance') if complete else None,card_state=card['state'],checks_passed=True,updated=now()))
    with (RUN/'HANDOFF.md').open('a',encoding='utf-8') as f:
        f.write('\n\n098оформлениезавершено:history_tracker_record/report.json/analysis.json. Научнаясерияполна='+str(bool(complete))+',карточка='+card['state']+'. '
            'Проверитьполныеreport/independent_audit/контроллеры/PIDend,частныеconfirmed_results/непройденныекритерии/ограничения. '
            '2эпохине6/40историческихval/неядроFlyVis/CPUголовасдопуском3e-5;точныйдискретныйповториспользуетпроверенныйGPUscore. '
            'После098НУЖЕНАГЕНТ:прочитатьисточники,сопоставитьсоткрытымипроблемами,выбратьобоснованноепродолжениевоставшемсяокне;незавершённоенеотрицательнаянаука.\n')


if __name__=='__main__':main()
