"""Complete failed recorder with correct evidence paths; no GPU or audit rerun."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,subprocess,shutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261029 import RUN,service

def main():
    import torch
    from evaluation.hota_eval import evaluate_hota
    torch.set_default_device('cpu');torch.set_num_threads(2)
    folder=RUN/'masa_final_checked_record'
    old=RUN/'masa_record_fp32_recovery1/controller_result.json'
    c=service.read(old);assert c['exit_code']==1 and c['all_owned_ended']
    assert not any(service.matching(k,v) for k,v in c['owned'].items())
    a=service.read(RUN/'masa_audit_fp32_recovery1/report.json')
    assert a['passed'] and a['complete'] and len(a['checked'])==20 and len(a['feature_checks'])==9
    assert max(x['max_difference'] for x in a['feature_checks'])<=1e-5
    controllers={}
    for name in ['masa_query_fp32_recovery1','masa_audit_fp32_recovery1']:
        c=service.read(RUN/name/'controller_result.json');assert c['exit_code']==0 and c['all_owned_ended']
        assert not any(service.matching(k,v) for k,v in c['owned'].items())
        controllers[name]=dict(sha256=service.sha(RUN/name/'controller_result.json'),all_owned_actually_ended=True)
    p=service.read(RUN/'masa_query_fp32_recovery1/protocol.json');videos=[x['video'] for x in p['sources']]
    assert len(set(videos))==20 and {x['video'] for x in a['checked']}==set(videos)
    pooled={}
    for name in p['variants']:
        service.write(folder/'status.json',dict(stage='Pooled HOTA on already audited outputs',variant=name,updated=service.now()))
        pooled[name]=evaluate_hota(RUN/'masa_audit_fp32_recovery1/metric_inputs'/name,videos,split='val')
    assert pooled['native']==pooled['zero']
    summary=dict(passed=True,complete=True,videos=20,trackers=5,pooled=pooled,paired=a['paired'],
        main_acceptance=a['main_acceptance'],full_discrete_audit_passed=True,
        feature_checks=a['feature_checks'],feature_max_difference=max(x['max_difference'] for x in a['feature_checks']),
        audit_sha256=service.sha(RUN/'masa_audit_fp32_recovery1/report.json'),
        old_record_failure_sha256=service.sha(old),controllers=controllers,
        relative_hota_percent={k:100*(v['HOTA']/pooled['native']['HOTA']-1) for k,v in pooled.items()},
        scientific_scope=a['scientific_scope'],limitations=p['limitations'],updated=service.now())
    service.write(folder/'scientific_summary.json',summary)
    h=service.read(ROOT/'experiments/project_history.json');card=next(x for x in h['experiments'] if x['id']=='reid_masa_detic_v1')
    shutil.copy2(ROOT/'experiments/project_history.json',folder/'project_history_before.json')
    src='runs/continuation_20261029/masa_final_checked_record/scientific_summary.json'
    paired=a['paired']
    values=dict(obtained=f'{src}:20историческихval/5вариантов,полныйдискретныйCPUаудит;9выборочныхencoderсверок,max={summary["feature_max_difference"]};pooled/pairedсодержатвсеHOTA/AssA/IDF1/FP/FN/IDSwисравнения. Detic-nativeсредняяпарнаяHOTA={paired["native"]["mean_per_video"]},95%ДИ={paired["native"]["ci95"]};Detic-R501333={paired["r50_1333"]["mean_per_video"]},95%ДИ={paired["r50_1333"]["ci95"]}.',
        interpretation='В одинаковой query-адаптации нашего полного трекера преимущество признаков MASA-Detic над исходным трекером и MASA-R50 при том же размере входа не подтверждено: оба доверительных интервала включают ноль. Это вывод об этой постановке, не опровержение официального MASA-Detic или идеи повторного узнавания объектов.',
        decision='Сохранить полный проверенный опыт. Более сильный извлекатель сам по себе не дал подтверждённой прибавки в данном правиле назначения; следующий шаг должен менять обоснованно локализованный узел, а не повторять20видео или подбирать порог по ним.',
        cannot_claim='Исторические20valранееиспользовалисьвБ1,не новый независимый итоговыйтест. Наша query-адаптация не полный официальный MASAтрекер;весаизвлекателейзаранеепредобучены,вес.1перенесёнсо старойdev-постановкибезподборанаval. НетвыводовобMGOR/ScaleTrack/ядреFlyVis. Весьдискретныйтрекерповторёнполностью;encoderперепроверенвыборочнона3заранеезаданныхвидео,некаждыйпиксель. ОбщиеFPN/fusion/headнеполнаянезависимаяMMDetectionреализация. Старыйraw-featureFP32gateостаётсянепройденным;новыйнормированныйинтерфейс1e-5пройденотдельно. ПервоначальнаяTF32серияисохранённыйаудиторскийотказнеподмененыFP32версией.')
    for key,value in values.items():
        if card.get(key):card.setdefault('revisions',[]).append(dict(date='2026-10-29',commit=None,field=key,old=card[key],reason='Полный проверенный FP32результат после сохранённой ошибки пути регистратора; прежние ограничения сохранены.'))
        card[key]=value
    card['state']='completed_positive' if a['main_acceptance'] else 'inconclusive';card['done']='2026-10-29'
    for i,item in enumerate(card['acceptance']):
        item['met']=True if i==0 else a['main_acceptance'];item['evidence']=src
    card['metrics']=[dict(key='HOTA',value=m['HOTA'],baseline=pooled['native']['HOTA'],subset='20historicalval;query;'+name,source=src,locator=['pooled',name,'HOTA']) for name,m in pooled.items()]
    card.setdefault('evidence',[]).append(dict(kind='file',ref=src,claim='Полный проверенный итог и все доверительные интервалы'))
    service.write(ROOT/'experiments/project_history.json',h)
    registry=ROOT/'experiments/tracking_hypotheses.json';shutil.copy2(registry,folder/'tracking_hypotheses_before.json');t=service.read(registry)
    method=next(x for x in t['methods'] if x['id']=='T10');assert method['status']=='planned'
    method.setdefault('history',[]).append(dict(date='2026-10-29',previous=method.copy(),reason='Прямое правило автора: завершённая проверка зелёная независимо от положительного исхода'))
    method['status']='completed'
    method['result']=dict(card_id=card['id'],source=src,sha256=service.sha(folder/'scientific_summary.json'),verdict='Неопределённый: преимущество не подтверждено',
        summary='Полный пилот20видео/5вариантовпроверен. Detic-native:средняяпарнаяHOTA+0,023901,95%ДИ[−0,006594;+0,063024];Detic-R501333:−0,000411,95%ДИ[−0,001973;+0,000811]. Обаинтервалачерезноль;этонашquery-адаптер,неполныйофициальныйMASAтрекер.')
    method['code_review']='Веса загружены строго, нормированный интерфейс проверен; полный FP32пилот20видео/5вариантов и независимый аудит завершены. Старые технические отказы сохранены в источниках.'
    service.write(registry,t)
    problems=service.read(ROOT/'experiments/system_problems.json');problem=next(x for x in problems['problems'] if x['id']=='С4')
    ev=dict(source=src,sha256=service.sha(folder/'scientific_summary.json'),claim='Более сильный extractor в одинаковом сопоставлении не дал подтверждённого прироста')
    problem['cases'].append(dict(date='2026-10-29',fact='Проверенный20-видеопилотMASA-DeticпротивR50содинаковым1333входом/назначениемнеподтвердилприбавку. Это ограничивает объяснение «достаточно усилить только признаки» в этой постановке; не доказывает общую причину проблем личности/памяти.',evidence=[ev]))
    problem['evidence'].append(ev);problem['updated']='2026-10-29';service.write(ROOT/'experiments/system_problems.json',problems)
    from scripts.b123_safe_journal_20261027 import journal
    journal(card['id'],'Полный проверенный результат MASA-Detic; Т10 опробована без подтверждённой пользы',card['obtained']+' '+card['interpretation']+' '+card['cannot_claim']+' Ошибка055путиmasa_query_audit/report.jsonсохранена,неповторялисьGPUилиаудит. Т10обновляетсязелёнымтолькоправилузавершения,другиегипотезынеизменяются.')
    commands=[['scripts/validate_project_history.py'],['-m','pytest','tests/test_project_history.py','tests/test_obsidian_tracking_hypotheses.py','tests/test_obsidian_fundamental_overview.py','-q'],['scripts/apply_tracking_completed_t10_20261029.py']]
    for i,args in enumerate(commands):
        with (folder/f'check_{i}.log').open('w',encoding='utf-8') as out:
            result=subprocess.run([sys.executable,*args],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,timeout=420).returncode
        assert result==0,(i,result)
    assert not torch.cuda.is_initialized()
    service.write(folder/'report.json',dict(complete=True,checks_passed=True,main_acceptance=a['main_acceptance'],summary=src,updated=service.now()))
    with (RUN/'HANDOFF.md').open('a',encoding='utf-8') as f:f.write('\n\n060masa_final_checked_recordполностьюоформилпроверенныйFP32итог20/5. Основныеисточникиscientific_summary.json/report.json/check_*.log. Главныйкритерийнепройден,Т10зелёнаякакпроверенная,снаучнымнеопределённымисходом. Б1–Б3/остальныеТиP1побайтны. 055ошибкапутисохранена,никакогоGPUповтораилиослабленияаудитаиз-заоформления. После060агентпроверяетcontroller/PIDend/всеисточники/граф,затемобоснованноепродолжениевразрешённомокне;неповторятьготовые20.\n')

if __name__=='__main__':main()
