"""Preserve the original record and explicitly state corrected negative contrasts."""
from pathlib import Path
import sys,subprocess,shutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261029 import RUN,service

def main():
    folder=RUN/'memory_write_agent_record'
    verified=service.read(RUN/'memory_write_agent_review/verified.json');assert verified['passed'] and verified['harm_vs_last_both_corrected']
    c=service.read(RUN/'memory_write_val_record/controller_result.json');assert c['exit_code']==0 and c['all_owned_ended']
    assert not any(service.matching(k,v) for k,v in c['owned'].items())
    src='runs/continuation_20261029/memory_write_val_audit/report.json';a=service.read(ROOT/src)
    h=service.read(ROOT/'experiments/project_history.json');card=next(x for x in h['experiments'] if x['id']=='query_memory_write_v1')
    shutil.copy2(ROOT/'experiments/project_history.json',folder/'project_history_before.json')
    text='Осторожное обновление памяти по уверенности детектора при beta=0,9 ухудшило HOTA относительно последнего наблюдения в обоих извлекателях. Средняя парная разница cautious−last: R50 −0,024644, интервал Bonferroni98,75% [−0,052100; −0,004834]; Detic −0,022980 [−0,050734; −0,003292]. Оба интервала целиком отрицательны. Относительно обычного экспоненциального усреднения оба скорректированных интервала включают ноль. Общий заранее заданный критерий пользы не пройден; положительных подтверждённых сравнений нет. Источник: '+src+'. Это отрицательный частный результат данной реализации записи, а не доказательство общей причины ошибок памяти.'
    decision='Эту реализацию усреднения по уверенности не продвигать как улучшение и не подбирать beta по этим val. Сохранить зарегистрированный общий исход inconclusive и явно отрицательные cautious−last сравнения. Рабочая альтернатива для отдельного согласования — решение о записи по неоднозначности назначения вместо одного балла детектора; сначала нужен различающий dev-протокол с согласованным контролем. Ошибочное смешивание личности и устаревание внешности пока не разделены измерением.'
    for key,value in [('interpretation',text),('decision',decision)]:
        card.setdefault('revisions',[]).append(dict(date='2026-10-29',commit=None,field=key,old=card[key],reason='Агент прочитал все четыре первичных сравнения: явно указано отрицательное направление вместо общего не подтверждено; критерии и исходы не изменены'))
        card[key]=value
    card['evidence'].append(dict(kind='file',ref='runs/continuation_20261029/memory_write_agent_review/verified.json',claim='Все двадцать SHA и фактический конец процессов проверены, оба ухудшения сохраняются после поправки'))
    service.write(ROOT/'experiments/project_history.json',h)
    problems=service.read(ROOT/'experiments/system_problems.json');problem=next(x for x in problems['problems'] if x['id']=='С5')
    problem['cases'].append(dict(date='2026-10-29',fact='Уточнение полного опыта записи памяти: '+text,evidence=[dict(source=src,sha256=service.sha(ROOT/src),claim='Оба cautious−last ухудшения после поправки на четыре сравнения')]))
    service.write(ROOT/'experiments/system_problems.json',problems)
    from scripts.b123_safe_journal_20261027 import journal
    journal(card['id'],'Агентская проверка: осторожная запись хуже последнего наблюдения с обоими извлекателями',text+' '+decision+' Первичные контроллеры110/120/125 и все20входных/выходныхSHA, готовые базовые результаты проверены;64tests161,22с/граф53видимых1361сохранённый/0удалений/0перемещений. Новогоencoderинференса/повторастарыхlast/обучениянебыло. ПоказателиобъединённойHOTAиmean_per_videoразличаютсяпоопределению;всеисходныечиславпервичномJSON.')
    for i,args in enumerate([['scripts/validate_project_history.py'],['-m','pytest','tests/test_project_history.py','tests/test_obsidian_fundamental_overview.py','tests/test_obsidian_tracking_hypotheses.py','-q']]):
        with (folder/f'check_{i}.log').open('w',encoding='utf-8') as out:r=subprocess.run([sys.executable,*args],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,timeout=420).returncode
        assert r==0,(i,r)
    from scripts.obsidian_fundamental_overview import build
    from scripts.audit_fundamental_obsidian import audit
    vault=ROOT/'ObsidianFlyVis';before={str(f.relative_to(vault)):service.sha(f) for f in vault.rglob('*') if f.is_file() and '.obsidian' not in f.parts}
    protected={f:h for f,h in before.items() if f=='Гипотезы.md' or f.startswith(('Гипотезы —','10 Проблемы мухи')) or (f.startswith('09 Запланировано') and Path(f).name.startswith(('Б1','Б2','Б3')))}
    copy=folder/'verified-vault';shutil.copytree(vault,copy);build(copy);g=audit(copy)
    assert g['passed'] and (g['hypothesis_node_count'],g['hypothesis_edge_count'],g['root_rays'])==(24,23,4)
    assert not g['external_hypothesis_edges'] and all((copy/f).is_file() for f in before)
    assert all(service.sha(copy/f)==h for f,h in protected.items())
    assert all(service.sha(vault/f)==h for f,h in before.items())
    build(vault);live=audit(vault);assert live['hashes']==g['hashes']
    assert all(service.sha(vault/f)==h for f,h in protected.items())
    service.write(folder/'graph_audit.json',dict(**live,deleted_files=0,moved_files=0,all_hypotheses_and_legacy_byte_identical=True))
    service.write(folder/'report.json',dict(complete=True,checks_passed=True,source=src,harm_vs_last_both_corrected=True,main_acceptance=False,updated=service.now()))

if __name__=='__main__':main()
