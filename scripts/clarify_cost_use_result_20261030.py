"""Clarify the verified cost result without changing criteria or numerical data."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import sys,subprocess,shutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261029 import RUN,service

def main():
    folder=RUN/'cost_use_agent_record';proof=service.read(RUN/'cost_use_agent_review/verified.json');assert proof['passed']
    c=service.read(RUN/'cost_use_val_record/controller_result.json')
    assert c['exit_code']==0 and c['all_owned_ended'] and not any(service.matching(k,v) for k,v in c['owned'].items())
    src='runs/continuation_20261029/cost_use_val_audit/report.json';a=service.read(ROOT/src)
    assert not a['main_acceptance']
    assert all(a['primary_comparisons'][e+'_old']['ci_global7'][0]>0 for e in ['r50','detic'])
    assert all(a['primary_comparisons'][e+'_shift']['ci_bonferroni_9875'][0]<=0<=a['primary_comparisons'][e+'_shift']['ci_bonferroni_9875'][1] for e in ['r50','detic'])
    assert all(x['ci_global7'][0]>0 for x in a['native_comparisons'].values())
    h=service.read(ROOT/'experiments/project_history.json');card=next(x for x in h['experiments'] if x['id']=='query_cost_use_v1')
    assert len(card['confirmed_results'])==5
    shutil.copy2(ROOT/'experiments/project_history.json',folder/'project_history_before.json')
    date=datetime.now(timezone(timedelta(hours=3))).date().isoformat()
    text='Изменение применения признаков улучшило прежнюю схему, но отдельная польза содержания внешности сверх общего сдвига не подтверждена. '
    for e,label in [('r50','R50'),('detic','Detic')]:
        old=a['primary_comparisons'][e+'_old'];shift=a['primary_comparisons'][e+'_shift']
        text+=f'{label}: средняя парная HOTA центрирование−прежняя добавка {old["mean_per_video"]:+.6f}, интервал Бонферрони98,75% {old["ci_bonferroni_9875"]}; центрирование−общий сдвиг {shift["mean_per_video"]:+.6f}, интервал {shift["ci_bonferroni_9875"]}. '
    text+='Оба первых интервала положительны; оба сравнения с общим сдвигом включают ноль. Это не доказательство равенства методов. '
    for n,label in [('r50_centered','Центрированная R50'),('detic_centered','Центрированная Detic'),('r50_shift','Общий сдвиг без содержания признаков')]:
        effect=a['native_comparisons'][n]
        text+=f'{label} относительно native: парная HOTA {effect["mean_per_video"]:+.6f}, дополнительный интервал семьи7 {effect["ci_global7"]}, положительный. '
    text+='Главный критерий всех четырёх сравнений не пройден; общий исход inconclusive. Сохранены пять частных положительных сравнений, включая вторичные native, прошедших более строгую поправку семьи7, записанную до dev; они не превращают главный исход в положительный. Источник: '+src+'. '
    text+='Локализовано влияние узла CostUseTracker.get_dists: при неизменных остальных условиях общий сдвиг стоимости сам даёт прибавку на этой выборке. Сам по себе этот опыт не устанавливает единственную корневую причину прежней слабой пользы внешности и не доказывает механизм только порога: меняется стоимость части допустимых пар внутри решателя.'
    decision='Сохранить центрирование и общий сдвиг как проверенные варианты авторской системы и отдельные частные результаты. Центрирование не принимать как доказанный способ извлечь дополнительную пользу содержания признаков: оно не превзошло контроль без внешности. Не подбирать центр/вес по этим двадцати видео. Для дальнейшего развития отделить калибровку стоимости от способности отличать личности; независимый итоговый тест проводить по отдельному заранее записанному протоколу, загрузку новых внешних данных согласовывать отдельно. В текущей разрешённой постановке все три новых трекера проверены; готовых новых научных заданий нет.'
    for key,value in [('interpretation',text),('decision',decision)]:
        card.setdefault('revisions',[]).append(dict(date=date,commit=None,field=key,old=card[key],reason='Агент прочитал все4primary/3secondary/pooled: отдельно записаны положительные частные эффекты и непройденный контроль содержания внешности'))
        card[key]=value
    service.write(folder/'confirmed_results_before_labels.json',card['confirmed_results'])
    for item in card['confirmed_results']:
        group,key=item['evidence']['locator']
        if group=='primary_comparisons':
            encoder='R50' if key.startswith('r50') else 'Detic'
            item['title']='Центрирование стоимости с '+encoder+' — сравнение с прежней добавкой'
            item['comparison']='Центрированная поправка с '+encoder+' против прежней положительной добавки с тем же извлекателем'
        elif key=='r50_shift':
            item['title']='Общий сдвиг стоимости без внешности — сравнение с исходным трекером'
            item['comparison']='Общий сдвиг без содержания внешности против исходного трекера'
        else:
            encoder='R50' if key.startswith('r50') else 'Detic'
            item['title']='Центрирование стоимости с '+encoder+' — сравнение с исходным трекером'
            item['comparison']='Центрированная поправка с '+encoder+' против исходного трекера'
    card['evidence'].append(dict(kind='file',ref='runs/continuation_20261029/cost_use_agent_review/verified.json',claim='Все двадцать SHA, входы, частные сравнения и фактический конец процессов проверены'))
    service.write(ROOT/'experiments/project_history.json',h)
    problems=service.read(ROOT/'experiments/system_problems.json');problem=next(x for x in problems['problems'] if x['id']=='С4')
    problem['cases'].append(dict(date=date,fact=text,evidence=[dict(source=src,sha256=service.sha(ROOT/src),claim='Контроль общего сдвига положителен к native, отдельное превосходство содержания внешности не подтверждено')]))
    problem['updated']=date;service.write(ROOT/'experiments/system_problems.json',problems)
    from scripts.b123_safe_journal_20261027 import journal
    journal(card['id'],'Проверка применения признаков: улучшение стоимости есть, вклад содержания не отделён',text+' '+decision+' Все20выходных/inputSHA/контроллеры/фактическиеPID+birthend проверены. Новогоencoder/GPU/обучения/пересчётаготовыхlastнет. Объединённая HOTA и средняя парная разница имеют разные определения; все метрики в первичномJSON.')
    for i,args in enumerate([['scripts/validate_project_history.py'],['-m','pytest','tests/test_project_history.py','tests/test_obsidian_fundamental_overview.py','tests/test_obsidian_tracking_hypotheses.py','-q']]):
        with (folder/f'check_{i}.log').open('w',encoding='utf-8') as log:r=subprocess.run([sys.executable,*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=420).returncode
        assert r==0,(i,r)
    from scripts.obsidian_fundamental_overview import build
    from scripts.audit_fundamental_obsidian import audit
    vault=ROOT/'ObsidianFlyVis';before={str(f.relative_to(vault)):service.sha(f) for f in vault.rglob('*') if f.is_file() and '.obsidian' not in f.parts}
    protected={f:v for f,v in before.items() if f=='Гипотезы.md' or f.startswith(('Гипотезы —','10 Проблемы мухи')) or (f.startswith('09 Запланировано') and Path(f).name.startswith(('Б1','Б2','Б3')))}
    copy=folder/'verified-vault';shutil.copytree(vault,copy);build(copy);g=audit(copy)
    assert g['passed'] and (g['hypothesis_node_count'],g['hypothesis_edge_count'],g['root_rays'])==(24,23,4)
    assert not g['external_hypothesis_edges'] and all((copy/f).is_file() for f in before)
    assert all(service.sha(copy/f)==v for f,v in protected.items()) and all(service.sha(vault/f)==v for f,v in before.items())
    build(vault);live=audit(vault);assert live['hashes']==g['hashes']
    assert all(service.sha(vault/f)==v for f,v in protected.items())
    service.write(folder/'graph_audit.json',dict(**live,deleted_files=0,moved_files=0,all_hypotheses_and_legacy_byte_identical=True))
    service.write(folder/'report.json',dict(complete=True,checks_passed=True,source=src,main_acceptance=False,confirmed_positive_comparisons=5,updated=service.now()))

if __name__=='__main__':main()
