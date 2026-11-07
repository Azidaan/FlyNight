"""Record the preregistered cost-use comparison, including the signal-free control."""
from pathlib import Path
import sys,subprocess,shutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261029 import RUN,service

def main():
    from datetime import datetime,timezone,timedelta
    date=datetime.now(timezone(timedelta(hours=3))).date().isoformat()
    folder=RUN/'cost_use_val_record'
    for n in ['cost_use_val','cost_use_val_audit']:
        c=service.read(RUN/n/'controller_result.json');assert c['exit_code']==0 and c['all_owned_ended']
        assert not any(service.matching(k,v) for k,v in c['owned'].items())
    src='runs/continuation_20261029/cost_use_val_audit/report.json';a=service.read(ROOT/src)
    assert a['complete'] and a['passed'] and a['audit_complete'] and len(a['checked'])==20
    assert len(a['primary_comparisons'])==4 and len(a['native_comparisons'])==3
    assert a['primary_comparisons']==service.read(RUN/'cost_use_val/report.json')['primary_comparisons']
    h=service.read(ROOT/'experiments/project_history.json');shutil.copy2(ROOT/'experiments/project_history.json',folder/'project_history_before.json')
    card=next(c for c in h['experiments'] if c['id']=='query_cost_use_v1')
    old_wins={e:a['primary_comparisons'][e+'_old']['ci_bonferroni_9875'][0]>0 for e in ['r50','detic']}
    signal_wins={e:a['primary_comparisons'][e+'_shift']['ci_bonferroni_9875'][0]>0 for e in ['r50','detic']}
    if a['main_acceptance']:
        interpretation='Центрированная поправка превзошла прежнюю добавку и общий сдвиг без содержания внешности с обоими извлекателями по всем четырём заранее заданным скорректированным интервалам. Это подтверждает пользу применения сигнала в данной постановке, не вклад ядра FlyVis.'
    else:
        interpretation='Общий критерий центрирования не пройден. Преимущество над прежней добавкой и над общим сдвигом проверяется раздельно; один выигрыш над прежней добавкой не доказывает пользу содержания признаков. Все четыре сравнения и неблагоприятные направления сохранены в первичном отчёте.'
    interpretation+=' Пройденные сравнения с прежней добавкой: '+str(old_wins)+'; с общим сдвигом: '+str(signal_wins)+'.'
    harmful={k:v for k,v in a['primary_comparisons'].items() if v['ci_bonferroni_9875'][1]<0}
    if harmful:interpretation+=' Явные частные ухудшения по скорректированным интервалам: '+str(harmful)+'. Их нельзя заменять формулировкой об отсутствии подтверждения.'
    updates=dict(obtained=src+': полные20историческихval/3новыхтрекера(centeredR50/centeredDetic/общийshift),старыеnative/old-lastпереиспользованыпоSHA. Весьraw/JSON/cost/write/метрики/pairedнезависимоповторён. ВсеHOTA/AssA/IDF1/FP/FN/IDSwвpooled,4primaryвprimary_comparisons,3secondarynativeвnative_comparisons,экстракторыописательновextractor_descriptive.',
        interpretation=interpretation,
        decision='Сохранить полный опыт. Перед продвижением сигнала проверить сравнение с общим сдвигом без признаков; самостоятельный эффект калибровки стоимости не приписывать повторному узнаванию или FlyVis. Для каждого следующего изменения требуется отдельная причинная мишень; готовые видео и старые входы не повторять.',
        cannot_claim='Это авторская query-адаптация, не полный Deep SORT/BoT-SORT и не ядро FlyVis. Память last,YOLO/признаки/допустимость геометрии прежние. Вес0,1/середина0,5 заданы до dev,без подбора по val. Те же20историческихval после последовательных исследований не являются независимым итоговым тестом. Признаки не пересчитывались; прежние9выборочныхencoderCPUchecks переиспользованы по SHA,это не полныйpixelreplay. Главный критерий использует4первичных сравнения/Bonferroni98,75%;частныеconfirmed_results допускаются только при дополнительно положительной нижней границе ci_global7 для заранее перечисленных4primary+3native. Сравнение извлекателей описательное и не входит в подтверждения.')
    for k,v in updates.items():
        if card.get(k):card.setdefault('revisions',[]).append(dict(date=date,commit=None,field=k,old=card[k],reason='Полный проверенный результат нового применения признаков; прежние поля сохранены'))
        card[k]=v
    card['state']='completed_positive' if a['main_acceptance'] else 'inconclusive';card['done']=date
    card['acceptance'][1].update(met=a['main_acceptance'],evidence=src)
    card['evidence'].append(dict(kind='file',ref=src,claim='Полный независимый результат, включая контроль без содержания признаков'))
    card['metrics']=[dict(key='HOTA',value=m['HOTA'],baseline=a['pooled']['native']['HOTA'],subset='20historicalval;cost-use;'+n,source=src,locator=['pooled',n,'HOTA']) for n,m in a['pooled'].items()]
    card['confirmed_results']=[]
    for group in ['primary_comparisons','native_comparisons']:
        for key,value in a[group].items():
            if value['ci_global7'][0]>0:
                card['confirmed_results'].append(dict(id='cost_use_'+group+'_'+key,title='Применение признаков — '+key,category='tracker',metric='HOTA',unit='points',
                    comparison=key,subset='20 исторических val',conditions='Заранее заданное сравнение; положительная нижняя ci_global7 для семьи4primary+3native, не только95%; память last и те же детекции/признаки/query.',
                    limitations=card['cannot_claim']+(' Вторичное сравнение с native, не главный критерий сигнала.' if group=='native_comparisons' else ''),
                    evidence=dict(source=src,sha256=service.sha(ROOT/src),locator=[group,key])))
    service.write(ROOT/'experiments/project_history.json',h)
    problems=service.read(ROOT/'experiments/system_problems.json');problem=next(x for x in problems['problems'] if x['id']=='С4')
    ev=dict(source=src,sha256=service.sha(ROOT/src),claim='Раздельная проверка содержания внешности и общего сдвига стоимости')
    problem['cases'].append(dict(date=date,fact=interpretation+' Само изменение HOTA не доказывает общую причину проблем системы.',evidence=[ev]));problem['evidence'].append(ev)
    service.write(ROOT/'experiments/system_problems.json',problems)
    from scripts.b123_safe_journal_20261027 import journal
    journal(card['id'],'Полная проверка применения признаков с контролем общего сдвига',card['obtained']+' '+interpretation+' '+str(a['primary_comparisons'])+' '+card['cannot_claim'])
    for i,args in enumerate([['scripts/validate_project_history.py'],['-m','pytest','tests/test_project_history.py','tests/test_query_cost_use.py','tests/test_obsidian_fundamental_overview.py','tests/test_obsidian_tracking_hypotheses.py','-q']]):
        with (folder/f'check_{i}.log').open('w',encoding='utf-8') as log:r=subprocess.run([sys.executable,*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=420).returncode
        assert r==0,(i,r)
    from scripts.obsidian_fundamental_overview import build
    from scripts.audit_fundamental_obsidian import audit
    vault=ROOT/'ObsidianFlyVis';before={str(f.relative_to(vault)):service.sha(f) for f in vault.rglob('*') if f.is_file() and '.obsidian' not in f.parts}
    protected={f:v for f,v in before.items() if f=='Гипотезы.md' or f.startswith(('Гипотезы —','10 Проблемы мухи')) or (f.startswith('09 Запланировано') and Path(f).name.startswith(('Б1','Б2','Б3')))}
    copy=folder/'verified-vault';shutil.copytree(vault,copy);build(copy);g=audit(copy)
    assert g['passed'] and (g['hypothesis_node_count'],g['hypothesis_edge_count'],g['root_rays'])==(24,23,4)
    assert not g['external_hypothesis_edges'] and all((copy/f).is_file() for f in before)
    assert all(service.sha(copy/f)==h for f,h in protected.items()) and all(service.sha(vault/f)==h for f,h in before.items())
    build(vault);live=audit(vault);assert live['hashes']==g['hashes']
    assert all(service.sha(vault/f)==h for f,h in protected.items())
    service.write(folder/'graph_audit.json',dict(**live,deleted_files=0,moved_files=0,all_hypotheses_and_legacy_byte_identical=True))
    service.write(folder/'report.json',dict(complete=True,checks_passed=True,main_acceptance=a['main_acceptance'],confirmed_result_count=len(card['confirmed_results']),source=src,updated=service.now()))
    (RUN/'HANDOFF.md').open('a',encoding='utf-8').write('\nПолныйcost-useопытзавершён/оформлен. АГЕНТ:прочитатьcost_use_val_audit/report.json/cost_use_val_record/report.json/контроллеры/PID+birthend/все20SHA/primary4/secondarynative3/описательноеextractor/confirmed_results/global7/граф. Отделитьобщийсдвиготпризнаков,пользуавторскойсистемыотядраFlyVis,негативныекритериинеудалять. Готовое20/encoder/oldlast/прежниеопытынеповторять.\n')

if __name__=='__main__':main()
