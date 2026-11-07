"""Record the completed paired decoder intervention and preserve graph history."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib, json, shutil, sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/flyvis_t4t5_paired_eval_recovery1_20261105'
sys.path.insert(0, str(ROOT))
from scripts.obsidian_system_problems import load, problem_path
from scripts.obsidian_fundamental_overview import build
from scripts.audit_fundamental_obsidian import audit

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p = json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    a = json.loads((OUT/'independent_audit.json').read_text(encoding='utf-8'))
    end = json.loads((OUT/'controller_result.json').read_text(encoding='utf-8'))
    assert a['passed'] and a['cases'] == 64 and a['protocol_sha256'] == sha(OUT/'protocol.json')
    assert a['inference_completed_sha256'] == sha(OUT/'inference_completed.json')
    assert end['exit_code'] == 0 and end['all_owned_ended']
    # The protocol compares non-convergence with reference only. The evaluator
    # additionally checks it against the other-cell control; record both.
    checks = []
    for ct in ('1.0', '0.25'):
        g = a['groups'][ct]
        checks.extend(a['paired'][f'{ct}_{k}']['upper'] < 0 for k in ('reference', 'control_other'))
        checks.extend(g['scale_t4']['censored_latency'] <= g[k]['censored_latency'] for k in ('reference', 'control_other'))
        checks.append(g['scale_t4']['not_found'] <= g['reference']['not_found'])
    primary = all(checks)
    refs = ['runs/flyvis_t4t5_paired_eval_recovery1_20261105/'+x for x in ('protocol.json','inference_completed.json','independent_audit.json','controller_result.json')]
    parts = []
    for ct in ('1.0', '0.25'):
        g = a['groups'][ct]
        def stat(k):
            x = g[k]
            return f"{x['epe']:.4f} (задержка с ограничением 24 кадра {x['censored_latency']:.3f}, без определения {x['not_found']}/32)"
        def pair(k):
            x=a['paired'][f'{ct}_{k}']
            return f"{x['mean']:+.4f} [{x['lower']:+.4f}; {x['upper']:+.4f}]"
        parts.append(f"Контраст {ct}: ошибка EPE, пикселя/кадр: штатная {stat('reference')}; T4 × 0,9 {stat('scale_t4')}; T5 × 0,9 {stat('scale_t5')}; четыре контрольных типа {stat('control_other')}. Парная разница T4 минус штатная {pair('reference')}, T4 минус контроль {pair('control_other')}, интервалы 98,75% с группировкой по восьми сценам. Фазовая корреляция без FlyVis: EPE {g['phase']['epe']:.4f}, некорректных переходов {g['phase']['invalid']}.")
    text=' '.join(parts)
    verdict='Выполнен' if primary else 'Не выполнен'
    limits=('64 зависимых искусственных условия построены из восьми ранее изученных изображений BDD TRAIN; четыре условия на сцену и контраст не являются независимыми сценами. Сравнение выполнено на одной штатной модели с искусственным общим переносом и разворотом, без настройки по этим результатам. Уменьшены копии сигналов непосредственно перед штатным декодером; рекуррентное состояние и параметры ядра не менялись. Контроль совпадает по числу типов, но не по величине сигнала или биологической роли. Нет реального видео движения, оценки трекера и HOTA; причинный вклад рекуррентной динамики не установлен. Фазовая корреляция оценена отдельно в благоприятной для общего переноса постановке.')
    discrepancy=('Число случаев без определения заранее требовалось не увеличивать относительно штатной модели; оценщик дополнительно проверил это относительно контроля других типов. Основной критерий ниже пересчитан по заранее записанному протоколу; дополнительная более строгая проверка сохранена в независимом отчёте.')
    msg=f'{text} Заранее заданный составной критерий: {verdict.lower()}. {discrepancy} {limits}'
    (OUT/'РЕЗУЛЬТАТ.md').write_text('# Проверка вмешательства в сигналы T4 и T5\n\n'+msg+'\n\nИсточники: '+', '.join('`'+x+'`' for x in refs)+'.\n', encoding='utf-8')
    cat=ROOT/'experiments/project_history.json'; d=json.loads(cat.read_text(encoding='utf-8')); c=next(x for x in d['experiments'] if x['id']==p['id']); assert c['state']=='planned'
    c.update(state='completed_positive' if primary else 'completed_negative',done=datetime.now(ZoneInfo('Europe/Moscow')).date().isoformat(),obtained=msg+' Источники: '+', '.join(refs),interpretation='Предварительная оценка влияния считываемых сигналов T4/T5 на ограниченном наборе искусственных разворотов; частные эффекты и ограничения приведены отдельно.',decision='Сохранить парный результат и заранее заданный критерий; обобщение проверять на новых сценах и реальном движении.',cannot_claim='Нет причинного доказательства роли рекуррентной динамики, изменения ядра, пользы трекеру или HOTA; восемь сцен ранее исследовались.')
    c['acceptance'][0].update(met=primary,evidence=refs[2]+': groups, paired; критерий записан в '+refs[0]); c['evidence'] += [dict(kind='json',ref=x,claim='Выполненная серия и независимая оценка') for x in refs[1:]]
    cat.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    prob=ROOT/'experiments/system_problems.json'; d=json.loads(prob.read_text(encoding='utf-8')); item=next(x for x in d['problems'] if x['id']=='С7'); item['cases'].append(dict(date='2026-11-05',fact=msg,evidence=refs)); item['cards'].append(c['id'])
    for x in refs: item['evidence'].append(dict(source=x,sha256=sha(ROOT/x),claim='Парная проверка влияния T4/T5 перед декодером'))
    prob.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    plan=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md'; raw=plan.read_bytes(); marker='## Журнал изменений'.encode(); nl=b'\r\n' if marker+b'\r\n' in raw else b'\n'; now=datetime.now(ZoneInfo('Europe/Moscow')); entry=f'\n**{now:%d.%m.%Y, %H:%M} MSK** — Парная проверка T4/T5 перед декодером\n\n{msg} Источники: '+', '.join('`'+x+'`' for x in refs)+'.\n\n'; assert raw.count(marker+nl)==1; plan.write_bytes(raw.replace(marker+nl,marker+nl+entry.encode(),1))
    with (ROOT/'FLYVIS_TRACKER_RESEARCH_ROADMAP.md').open('a',encoding='utf-8') as f: f.write('\n\n## 05.11.2026 — парное вмешательство в сигналы T4/T5\n\n'+msg+' Источник: `'+refs[2]+'`.\n')
    vault=ROOT/'ObsidianFlyVis'; stage=OUT/'vault_checked'; backup=OUT/'vault_before'; assert not stage.exists() and not backup.exists(); shutil.copytree(vault,stage); shutil.copytree(vault,backup)
    before={f.relative_to(vault).as_posix():sha(f) for f in vault.rglob('*') if f.is_file()}; name=problem_path(next(x for x in load()['problems'] if x['id']=='С7'))+'.md'; preserve=[x for x in before if x.endswith('.md') and x!=name]; assert build(stage,preserve)==0
    for f in ('.obsidian/graph.json','.graph-visible.json','.generated.json'): shutil.copy2(backup/f,stage/f)
    check=audit(stage); assert check['note_count']==55 and check['hypothesis_node_count']==25 and check['problem_node_count']==10 and not check['external_hypothesis_edges']; assert set(before)=={f.relative_to(stage).as_posix() for f in stage.rglob('*') if f.is_file()}; changed=[f for f,h in before.items() if sha(stage/f)!=h]; assert changed==[name] and all(sha(vault/f)==h for f,h in before.items()); shutil.copy2(stage/name,vault/name); done=audit(vault); assert done['hashes']==check['hashes']; done.update(changed_files=changed,added_files=[],deleted_files=[],protected_hypotheses_byte_exact=True); (OUT/'graph_audit.json').write_text(json.dumps(done,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print('Recorded and graph checked; protocol criterion:',primary)
if __name__=='__main__': main()
