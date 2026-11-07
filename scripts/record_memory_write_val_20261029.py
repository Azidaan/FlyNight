"""Save only a complete accepted independent memory-policy evaluation."""
from pathlib import Path
import sys,subprocess,shutil
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261029 import RUN,service

def main():
    folder=RUN/'memory_write_val_record'
    for name in ['memory_write_val','memory_write_val_audit']:
        c=service.read(RUN/name/'controller_result.json');assert c['exit_code']==0 and c['all_owned_ended']
        assert not any(service.matching(k,v) for k,v in c['owned'].items())
    a=service.read(RUN/'memory_write_val_audit/report.json')
    assert a['passed'] and a['complete'] and a['audit_complete'] and len(a['checked'])==20
    assert a['primary_comparisons']==service.read(RUN/'memory_write_val/report.json')['primary_comparisons']
    h=service.read(ROOT/'experiments/project_history.json');shutil.copy2(ROOT/'experiments/project_history.json',folder/'project_history_before.json')
    card=next(c for c in h['experiments'] if c['id']=='query_memory_write_v1')
    src='runs/continuation_20261029/memory_write_val_audit/report.json'
    updates=dict(obtained=f'{src}: 20 исторических val, четыре новых варианта памяти, полное повторение raw/JSON/событий стоимости/векторов записи памяти/метрик. Старые native/last и точные GPU FP32-векторы переиспользованы по SHA; нового извлечения признаков не было. Все HOTA/AssA/IDF1/FP/FN/IDSw в pooled, четыре парных сравнения с обычными и скорректированными интервалами в primary_comparisons.',
        interpretation=('Осторожное обновление памяти превзошло последнюю запись и обычное усреднение в обоих извлекателях по всем четырём заранее заданным скорректированным интервалам.' if a['main_acceptance'] else 'Общее преимущество осторожного обновления над последней записью и обычным усреднением в обоих извлекателях не подтверждено по всем четырём заранее заданным скорректированным интервалам. Частные результаты сохранены отдельно.'),
        decision='Сохранить опыт без повторения готовых видео и подбора параметров на них. Сопоставить направление изменений с проблемой записи памяти; причина загрязнения личности не считается доказанной одной разницей HOTA. Следующая правка требует конкретного различающего механизма.',
        cannot_claim='Это адаптация отдельного правила записи, не воспроизведение полного Deep OC-SORT и не изменение ядра FlyVis. Уверенность детектора не является меткой личности; медленное обновление может закреплять ошибку. Beta0,9 задана до dev, качество dev не использовалось для выбора варианта. Те же двадцать исторических val не являются новым независимым итоговым тестом. Прежние GPU-векторы переиспользованы точно; их независимая выборочная проверка в девяти точках переиспользована по SHA, это не полный повтор всех пикселей.')
    for k,v in updates.items():
        if card.get(k):card.setdefault('revisions',[]).append(dict(date='2026-10-29',commit=None,field=k,old=card[k],reason='Полный независимый результат памяти; прежние ограничения сохраняются'))
        card[k]=v
    card['state']='completed_positive' if a['main_acceptance'] else 'inconclusive';card['done']='2026-10-29'
    card['acceptance'][1].update(met=a['main_acceptance'],evidence=src)
    card['evidence'].append(dict(kind='file',ref=src,claim='Все двадцать видео независимо проверены; опубликованы неблагоприятные исходы'))
    native=a['pooled']['native']['HOTA']
    card['metrics']=[dict(key='HOTA',value=m['HOTA'],baseline=native,subset='20historicalval;memory-write;'+n,source=src,locator=['pooled',n,'HOTA']) for n,m in a['pooled'].items()]
    card['confirmed_results']=[]
    for key,value in a['primary_comparisons'].items():
        if value['ci_bonferroni_9875'][0]>0:
            card['confirmed_results'].append(dict(id='memory_write_'+key,title='Осторожная запись памяти против '+key,category='tracker',metric='HOTA',unit='points',comparison=key,subset='20 исторических val',
                conditions='Те же признаки, детекции и назначения; beta0,9; положительна также нижняя граница Bonferroni98,75%',limitations=card['cannot_claim'],
                evidence=dict(source=src,sha256=service.sha(ROOT/src),locator=['primary_comparisons',key])))
    service.write(ROOT/'experiments/project_history.json',h)
    problems=service.read(ROOT/'experiments/system_problems.json');c=next(c for c in problems['problems'] if c['id']=='С5')
    ev=dict(source=src,sha256=service.sha(ROOT/src),claim='Полная согласованная проверка записи памяти; сама причина загрязнения личности отдельно не измерена')
    c['cases'].append(dict(date='2026-10-29',fact=card['interpretation']+' Это не доказательство общей причины ошибок истории.',evidence=[ev]))
    c['evidence'].append(ev);c['updated']='2026-10-29';service.write(ROOT/'experiments/system_problems.json',problems)
    from scripts.b123_safe_journal_20261027 import journal
    journal(card['id'],'Полная независимая проверка записи памяти на исторических val',card['obtained']+' '+card['interpretation']+' '+str(a['primary_comparisons'])+' '+card['cannot_claim'])
    commands=[['scripts/validate_project_history.py'],['-m','pytest','tests/test_project_history.py','tests/test_query_memory_write.py','tests/test_obsidian_fundamental_overview.py','tests/test_obsidian_tracking_hypotheses.py','-q']]
    for i,args in enumerate(commands):
        with (folder/f'check_{i}.log').open('w',encoding='utf-8') as output:r=subprocess.run([sys.executable,*args],cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,timeout=420).returncode
        assert r==0,(i,r)
    from scripts.obsidian_fundamental_overview import build
    from scripts.audit_fundamental_obsidian import audit
    vault=ROOT/'ObsidianFlyVis';before={str(f.relative_to(vault)):service.sha(f) for f in vault.rglob('*') if f.is_file() and '.obsidian' not in f.parts}
    protected={f:h for f,h in before.items() if f=='Гипотезы.md' or f.startswith(('Гипотезы —','10 Проблемы мухи')) or (f.startswith('09 Запланировано') and Path(f).name.startswith(('Б1','Б2','Б3')))}
    copy=folder/'verified-vault';shutil.copytree(vault,copy)
    build(copy);g=audit(copy);assert g['passed'] and (g['hypothesis_node_count'],g['hypothesis_edge_count'],g['root_rays'])==(24,23,4)
    assert not g['external_hypothesis_edges']
    assert all((copy/f).exists() for f in before)
    assert all(service.sha(copy/f)==h for f,h in protected.items())
    assert all(service.sha(vault/f)==h for f,h in before.items())
    build(vault);live=audit(vault);assert live['hashes']==g['hashes']
    assert all(service.sha(vault/f)==h for f,h in protected.items())
    service.write(folder/'graph_audit.json',dict(**live,deleted_files=0,moved_files=0,all_other_hypotheses_and_legacy_byte_identical=True))
    service.write(folder/'report.json',dict(complete=True,checks_passed=True,main_acceptance=a['main_acceptance'],source=src,graph_checked=True,updated=service.now()))
    (RUN/'HANDOFF.md').open('a',encoding='utf-8').write('\nПолная проверка записи памяти завершена и оформлена. memory_write_val/independent_audit.json,memory_write_val_audit/report.json,memory_write_val_record/report.json/graph_audit.json/check_*.log. АГЕНТ обязан прочитать полные primary/natives/pooled/контроллеры/PID+birthend и частные confirmed_results, затем научный вывод. Не повторять готовые двадцать/encoder/last. Б2/GPU-аудит не принят; новая память не является ядром FlyVis или полным DeepOC-SORT.\n')

if __name__=='__main__':main()
