"""Записать посмертный разбор в существующий случай С7 и журнал."""
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import hashlib,json,shutil,sys

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/flyvis_tau_half_components_20261104'
sys.path.insert(0,str(ROOT))
from scripts.obsidian_system_problems import load,problem_path
from scripts.obsidian_fundamental_overview import build
from scripts.audit_fundamental_obsidian import audit


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    r=json.loads((OUT/'report.json').read_text(encoding='utf-8'))
    a=json.loads((OUT/'independent_audit.json').read_text(encoding='utf-8'))
    assert r['complete'] and a['passed'] and a['report_sha256']==sha(OUT/'report.json')
    text=('Дополнительный описательный разбор тех же сохранённых результатов: при обычном контрасте '
        'до разворота средняя номинальная ошибка движения выросла с 2,0092 до 2,3751; '
        'за первые пять кадров после разворота уменьшилась с 2,2219 до 1,9613, '
        'а за последующие 19 кадров выросла с 1,4667 до 1,6986. '
        'Среднее отношение установившегося отклика с вычетом неподвижного уровня к заданной скорости '
        'уменьшилось с 0,3921 до 0,2002; это признак ослабления сигнала в данной постановке, '
        'но не доказанная причина на уровне отдельных нейронов. После остановки средний модуль '
        'остатка вдоль движения после вычета неподвижного уровня за 24 кадра уменьшился с 0,1473 '
        'до 0,0854. Значения описательные, получены на прежних четырёх изображениях и одной модели; '
        'выход не калиброван, новые данные и HOTA не проверены.')
    doc=('# Из чего сложилась ошибка после ускорения FlyVis\n\n'+text+'\n\n'
         'Отклик после смены направления улучшается в первые пять кадров, но при продолжающемся '
         'движении усреднённый сигнал становится слабее относительно заданного перемещения. '
         'Сохранённый результат позволяет описать этот обмен, но не устанавливает единственный '
         'внутренний механизм. Временные окна выбраны после опыта; вывод нужно проверять отдельно.\n\n'
         'Квадрат ошибки разложен по направлению движения и поперёк него. Средние абсолютные '
         'ошибки по осям не складываются в обычную среднюю векторную ошибку. '
         'Источники: `report.json`, `independent_audit.json`, '
         '`../flyvis_tau_half_20261104/report.json`.\n')
    (OUT/'РЕЗУЛЬТАТ.md').write_text(doc,encoding='utf-8')
    path=ROOT/'experiments/system_problems.json'
    shutil.copy2(path,OUT/'system_problems_before.json')
    data=json.loads(path.read_text(encoding='utf-8'))
    p=next(x for x in data['problems'] if x['id']=='С7')
    files=['report.json','independent_audit.json']
    sources=['runs/flyvis_tau_half_components_20261104/'+name for name in files]
    p['cases'].append(dict(date='2026-11-04',fact=text,evidence=sources))
    p['evidence'] += [dict(source=s,sha256=sha(OUT/name),claim=claim) for s,name,claim in zip(sources,files,
       ['Разбор направления, величины и остатка после остановки','Независимая проверка вычислений по сохранённым векторам'])]
    p['hypotheses'].append('Рабочая версия после описательного разбора: уменьшение масштаба отклика при движении могло внести вклад в рост номинальной ошибки; причинный механизм и перенос на новые сцены не установлены.')
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    plan=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';raw=plan.read_bytes();mark='## Журнал изменений'.encode();nl=b'\r\n' if mark+b'\r\n' in raw else b'\n'
    now=datetime.now(ZoneInfo('Europe/Moscow'))
    entry=f'\n**{now:%d.%m.%Y, %H:%M} MSK** — Разбор ошибки ускоренной FlyVis\n\n{text} Источники `{sources[0]}` и `{sources[1]}`. Следствие: проблема С7 открыта, причина на уровне отдельных нейронов не доказана; новые эксперименты не запускались.\n\n'
    assert raw.count(mark+nl)==1
    plan.write_bytes(raw.replace(mark+nl,mark+nl+entry.encode(),1))
    vault=ROOT/'ObsidianFlyVis';backup=OUT/'vault_before';stage=OUT/'vault_checked'
    assert not backup.exists() and not stage.exists()
    shutil.copytree(vault,backup);shutil.copytree(vault,stage)
    before={p.relative_to(vault).as_posix():sha(p) for p in vault.rglob('*') if p.is_file()}
    name=problem_path(next(x for x in load()['problems'] if x['id']=='С7'))+'.md'
    preserve=[x for x in before if x.endswith('.md') and x!=name]
    assert build(stage,preserve)==0
    for f in ['.obsidian/graph.json','.graph-visible.json','.generated.json']:shutil.copy2(backup/f,stage/f)
    checked=audit(stage)
    assert checked['note_count']==55 and checked['hypothesis_node_count']==25 and checked['problem_node_count']==10
    assert not checked['external_hypothesis_edges']
    assert set(before)=={p.relative_to(stage).as_posix() for p in stage.rglob('*') if p.is_file()}
    changes=[x for x,h in before.items() if sha(stage/x)!=h]
    assert changes==[name],changes
    assert all(sha(vault/x)==h for x,h in before.items())
    shutil.copy2(stage/name,vault/name)
    done=audit(vault);assert done['hashes']==checked['hashes']
    done.update(changed_files=changes,added_files=[],deleted_files=[],moved_files=[],protected_hypotheses_byte_exact=True)
    (OUT/'graph_audit.json').write_text(json.dumps(done,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=done['passed'],changed=changes),ensure_ascii=False))


if __name__=='__main__':main()
