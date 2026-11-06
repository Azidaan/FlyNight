"""Record this bounded diagnostic without changing protected hypothesis nodes."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json,hashlib,shutil,sys
import psutil
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/flyvis_history_state_swap_20261106'
sys.path.insert(0,str(ROOT))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    from scripts.flyvis_history_state_swap_20261106 import journal
    a=json.loads((OUT/'independent_audit.json').read_text(encoding='utf-8'));end=json.loads((OUT/'controller_result.json').read_text());assert a['passed'] and a['cases']==32 and end['exit_code']==0 and end['all_owned_ended']
    launch=json.loads((OUT/'controller.launch.json').read_text());owned={**end['owned'],str(launch['pid']):launch['birth']}
    for pid,birth in owned.items():
        try:assert abs(psutil.Process(int(pid)).create_time()-birth)>.01, f'Owned PID still alive: {pid}'
        except psutil.NoSuchProcess:pass
    (OUT/'ended_verified.json').write_text(json.dumps(dict(checked_msk=datetime.now(ZoneInfo('Europe/Moscow')).isoformat(),all_owned_ended=True,owned=owned),indent=2)+'\n',encoding='utf-8')
    refs=[str(OUT.relative_to(ROOT)/f).replace('\\','/') for f in ['protocol.json','inference_completed.json','independent_audit.json','controller_result.json','ended_verified.json']]
    text=f"На всех {a['cases']} условиях полная двусторонняя подмена состояния и тождественные контроли дали точное совпадение состояний и выходов с соответствующим донором. Предыстория изменила выход выше допуска 1e-5 в {a['history_effect_cases']} случаях."
    for contrast,g in a['groups'].items():
        x,y=g['moving'],g['stationary'];ci=g['stationary_minus_moving_epe']['ci95']
        text+=f" Контраст {contrast}: после движения / после покоя ошибка {x['epe']:.6f} / {y['epe']:.6f} номинального пикселя на кадр; задержка с ограничением 24 кадра {x['capped_latency']:.4f} / {y['capped_latency']:.4f}; без определения {x['not_found']}/16 / {y['not_found']}/16. Парная разница ошибки покой минус движение {g['stationary_minus_moving_epe']['mean']:+.6f}, описательный 95% интервал [{ci[0]:+.6f}; {ci[1]:+.6f}] по четырём сценам. Изменение ошибки к движущейся предыстории {(y['epe']/x['epe']-1)*100:+.3f}%."
    interpretation='При одинаковом продолжении влияние предыстории переносится полным рекуррентным состоянием ядра. Само точное воспроизведение донора — контроль исправности, а не доказательство полезного исправления. Различие включает адаптацию к предыстории; конкретная группа клеток и механизм вредного запаздывания не локализованы.'
    limits='Одна модель, четыре ранее изученных изображения и 32 зависимых искусственных условия; это не новый независимый тест. Состояние неподвижного донора построено специально и не является доступным универсальным сбросом в реальном видео. Нет обучения, новой калибровки, HOTA и доказательства решения проблемы задержки. Доля измеренного потолка не определена.'
    decision='Сохранить проверенный способ полного переноса. Не внедрять неподвижный сброс как исправление. Следующее вмешательство следует ограничить согласованной группой состояний, с полным переносом как положительным контролем; выбор группы требует отдельного обоснования.'
    msg=text+' Источник чисел: '+refs[2]+', groups и rows.'
    (OUT/'РЕЗУЛЬТАТ.md').write_text('# FlyVis — влияние предыстории и полная подмена состояния\n\n'+msg+'\n\n'+interpretation+'\n\n'+limits+'\n\n'+decision+'\n\nИсточники: '+', '.join('`'+r+'`' for r in refs)+'.\n',encoding='utf-8')
    cat=ROOT/'experiments/project_history.json';d=json.loads(cat.read_text(encoding='utf-8'));c=next(x for x in d['experiments'] if x['id']=='flyvis_history_state_swap_v1');assert c['state']=='planned'
    c.update(state='inconclusive',done=datetime.now(ZoneInfo('Europe/Moscow')).date().isoformat(),obtained=msg,interpretation=interpretation,decision=decision,cannot_claim=limits,confirmed_results=[])
    c['acceptance'][0].update(met=True,evidence=refs[2]);c['evidence'] += [dict(kind='json',ref=r,claim='Результат и независимая проверка') for r in refs[1:]]
    cat.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    prob=ROOT/'experiments/system_problems.json';d=json.loads(prob.read_text(encoding='utf-8'));item=next(x for x in d['problems'] if x['id']=='С7')
    item['cases'].append(dict(date='2026-11-06',fact=msg+' '+interpretation+' '+limits,evidence=refs));item['cards'].append(c['id'])
    item['evidence'] += [dict(source=r,sha256=sha(ROOT/r),claim='Полная подмена состояния и одинаковое продолжение') for r in refs]
    prob.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');journal(msg+' '+interpretation+' '+limits+' '+decision)
    with (ROOT/'FLYVIS_TRACKER_RESEARCH_ROADMAP.md').open('a',encoding='utf-8') as f:f.write('\n\n## 06.11.2026 — подмена полного состояния ядра\n\n'+msg+' '+interpretation+' '+limits+'\n')
    from scripts.obsidian_system_problems import load,problem_path
    from scripts.obsidian_fundamental_overview import build
    from scripts.audit_fundamental_obsidian import audit
    vault=ROOT/'ObsidianFlyVis';stage=OUT/'vault_checked';backup=OUT/'vault_before'
    assert not stage.exists() and not backup.exists();shutil.copytree(vault,stage);shutil.copytree(vault,backup)
    before={f.relative_to(vault).as_posix():sha(f) for f in vault.rglob('*') if f.is_file()};name=problem_path(next(x for x in load()['problems'] if x['id']=='С7'))+'.md'
    preserve=[f for f in before if f.endswith('.md') and f!=name];assert build(stage,preserve)==0
    for f in ('.obsidian/graph.json','.graph-visible.json','.generated.json'):shutil.copy2(backup/f,stage/f)
    check=audit(stage);assert not check['external_hypothesis_edges']
    assert set(before)=={f.relative_to(stage).as_posix() for f in stage.rglob('*') if f.is_file()}
    changed=[f for f,h in before.items() if sha(stage/f)!=h];assert changed==[name]
    assert all(sha(vault/f)==h for f,h in before.items()), 'Concurrent vault edit: do not overwrite'
    shutil.copy2(stage/name,vault/name);done=audit(vault);assert done['hashes']==check['hashes'];done.update(changed_files=changed,deleted_files=[],protected_hypotheses_byte_exact=True)
    (OUT/'graph_audit.json').write_text(json.dumps(done,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(msg)
if __name__=='__main__':main()
