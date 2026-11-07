"""Preserve complete favorable and unfavorable partial-transplant findings."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json,hashlib,shutil,sys
import psutil
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/flyvis_partial_state_swap_20261106';sys.path.insert(0,str(ROOT))
LABELS=dict(retina='Входные R1–R8',intermediate='Промежуточные клетки, не считываемые напрямую',t4='T4',t5='T5',other_readout='Остальные считываемые клетки',t4t5='T4 и T5 вместе')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    from scripts.flyvis_partial_state_swap_20261106 import journal
    p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'));a=json.loads((OUT/'independent_audit.json').read_text(encoding='utf-8'));end=json.loads((OUT/'controller_result.json').read_text());launch=json.loads((OUT/'controller.launch.json').read_text())
    assert a['passed'] and a['cases']==32 and a['protocol_sha256']==sha(OUT/'protocol.json') and end['exit_code']==0 and end['all_owned_ended']
    owned={**end['owned'],str(launch['pid']):launch['birth']}
    for pid,birth in owned.items():
        try:assert abs(psutil.Process(int(pid)).create_time()-birth)>.01,f'Own process still alive: {pid}'
        except psutil.NoSuchProcess:pass
    now=datetime.now(ZoneInfo('Europe/Moscow'));(OUT/'ended_verified.json').write_text(json.dumps(dict(checked_msk=now.isoformat(),all_owned_ended=True,owned=owned),indent=2)+'\n',encoding='utf-8')
    refs=[str(OUT.relative_to(ROOT)/f).replace('\\','/') for f in ['protocol.json','inference_completed.json','independent_audit.json','controller_result.json','ended_verified.json']]
    accepted=[g for g,v in a['candidates'].items() if v['passed']]
    intro='Проверены шесть заранее заданных групп, каждая в двух направлениях переноса состояния. Для каждого из 32 условий проверены разворот и продолжение движения. Все контроли копирования и независимый пересчёт пройдены. '+('Критерий сохранения полезного ответа прошли: '+', '.join(LABELS[g] for g in accepted)+'.' if accepted else 'Ни одна группа не прошла одновременно критерии уменьшения задержки и сохранения продолжающегося движения.')
    paragraphs=[];brief=[]
    for g in p['groups']:
        parts=[]
        for contrast in ('1.0','0.25'):
            r=a['groups'][contrast]['reverse'];cont=a['groups'][contrast]['continue'];x=r['erase_'+g];base=r['moving'];ci=a['paired'][f'{contrast}:reverse:erase_{g}:capped_latency']['ci_family6' if contrast=='1.0' else 'ci95']
            interval_label='99,1667% интервал с поправкой на шесть проверок' if contrast=='1.0' else 'описательный 95% интервал'
            parts.append(f"Контраст {contrast}: задержка после подмены {x['capped_latency']:.4f} против исходной {base['capped_latency']:.4f} кадра; без определения {x['not_found']}/16 против {base['not_found']}/16. Разница задержки {x['capped_latency']-base['capped_latency']:+.4f}, {interval_label} [{ci[0]:+.4f}; {ci[1]:+.4f}]. Ошибка разворота {x['epe']:.6f} против {base['epe']:.6f} ({(x['epe']/base['epe']-1)*100:+.3f}%). Ошибка продолжающегося движения за все 24 кадра {cont['erase_'+g]['epe']:.6f} против {cont['moving']['epe']:.6f}, за первые шесть кадров {cont['erase_'+g]['early_epe']:.6f} против {cont['moving']['early_epe']:.6f}. Обратный перенос движущегося состояния в неподвижное дал задержку {r['insert_'+g]['capped_latency']:.4f} против {r['stationary']['capped_latency']:.4f} кадра.")
        failures=[k for k,v in a['candidates'][g]['checks'].items() if not v]
        paragraphs.append(LABELS[g]+'. '+' '.join(parts)+' Критерий: '+('пройден.' if not failures else 'не пройден; невыполненные условия: '+', '.join(failures)+'.'))
        r=a['groups']['1.0']['reverse'];cont=a['groups']['1.0']['continue'];brief.append(f"{LABELS[g]}: задержка {r['erase_'+g]['capped_latency']:.4f} вместо {r['moving']['capped_latency']:.4f}; ошибка первых шести кадров продолжающегося движения {cont['erase_'+g]['early_epe']:.6f} вместо {cont['moving']['early_epe']:.6f}")
    reverse=a['groups']['1.0']['reverse'];continuing=a['groups']['1.0']['continue']
    localization=(f"Наибольшее сокращение задержки среди проверенных групп дала подмена {len(p['groups']['intermediate'])} промежуточных типов, не считываемых декодером напрямую: {reverse['moving']['capped_latency']:.4f} → {reverse['erase_intermediate']['capped_latency']:.4f} кадра. Обратный перенос увеличил задержку {reverse['stationary']['capped_latency']:.4f} → {reverse['insert_intermediate']['capped_latency']:.4f}. При продолжающемся движении ошибка первых шести кадров выросла на {(continuing['erase_intermediate']['early_epe']/continuing['moving']['early_epe']-1)*100:.3f}%. Группа включает "+', '.join(p['groups']['intermediate'])+'. Это локализация переносимого состояния до большой группы, а не установление единственного нейрона или места, где это состояние формируется.')
    interpretation=localization+' Это вмешательство в рекуррентное состояние, а не только в вход декодера. Изолированное уменьшение задержки не доказывает, что группа является единственной причиной проблемы: частично заменённое состояние может быть несогласовано с остальной сетью. Обратная подмена и полные контроли ограничивают объяснение, но не превращают искусственный донор в готовое исправление.'
    limits='Одна модель и четыре ранее изученных изображения BDD TRAIN, искусственное движение, 32 зависимых условия; не новый независимый итоговый тест. Контроль продолжающегося движения проверяет только одну функцию памяти. Группы отличаются числом клеток и величиной переносимой активности. Нулевая ширина интервала для промежуточной группы вызвана одинаковыми четырьмя сценными разностями и дискретностью задержки, а не отсутствием неопределённости вне этих сцен. Нет доказательства сохранения всей полезной памяти, улучшения реального трекинга или HOTA. Доля измеренного потолка не определена.'
    decision=('Сохранить прошедшие группы только как кандидаты для отдельной реализуемой правки; ничего не включать в штатный путь без новых контролей.' if accepted else 'Не внедрять стирание или подмену проверенных групп. Сохранить обнаруженные размены между задержкой и полезным ответом; не подбирать новые группы или силу вмешательства по этим изображениям.')
    literature='Литературное основание: Eyal Gruntman, Pablo Reimers, Sandro Romani, Michael B. Reiser (2021), «Non-preferred contrast responses in the Drosophila motion pathways reveal a receptive field structure that explains a common visual illusion», [оригинал статьи](https://doi.org/10.1016/j.cub.2021.09.072). Работа о различиях ответов путей T4/T5 помогла заранее выбрать отдельную и совместную подмены этих групп. Это не опубликованное решение нашего запаздывания. Архитектура рекуррентной модели и отдельного считывателя: Lappalainen и соавторы (2024), «Connectome-constrained networks predict neural activity across the fly visual system», [оригинал статьи](https://doi.org/10.1038/s41586-024-07939-3). Источники и конкретная роль зафиксированы в протоколе.'
    full=intro+'\n\n'+'\n\n'.join(paragraphs)+'\n\n'+interpretation+'\n\n'+limits+'\n\n'+decision+'\n\n'+literature
    (OUT/'РЕЗУЛЬТАТ.md').write_text('# Части состояния FlyVis: задержка и полезная память\n\n'+full+'\n\nИсточник всех чисел: `'+refs[2]+'`, groups, paired, candidates и rows.\n',encoding='utf-8')
    cat=ROOT/'experiments/project_history.json';d=json.loads(cat.read_text(encoding='utf-8'));c=next(x for x in d['experiments'] if x['id']==p['id']);assert c['state']=='planned'
    c.update(state='completed_positive' if accepted else 'completed_negative',done=now.date().isoformat(),obtained=intro+' '+'; '.join(brief)+'. Источник: '+refs[2]+', groups; полные результаты '+str(OUT.relative_to(ROOT)/'РЕЗУЛЬТАТ.md'),interpretation=interpretation,decision=decision,cannot_claim=limits,confirmed_results=[])
    c['acceptance'][0].update(met=bool(accepted),evidence=refs[2]+': candidates; все технические проверки passed=true.');c['evidence'] += [dict(kind='json',ref=r,claim='Завершённая проверка и первичный результат') for r in refs[1:]]
    # Preserve partial positive contrasts, explicitly paired with their harmful cost.
    partials=dict(source=refs[2],source_sha256=sha(OUT/'independent_audit.json'),effects={})
    for g in p['groups']:
        pair=a['paired'][f'1.0:reverse:erase_{g}:capped_latency']
        if pair['ci_family6'][1]<0:
            partials['effects'][g]=dict(mean_per_video=-pair['mean'],ci95=[-pair['ci95'][1],-pair['ci95'][0]],ci_corrected=[-pair['ci_family6'][1],-pair['ci_family6'][0]],n=4,baseline=reverse['moving']['capped_latency'],early_continuation_error_percent=100*(continuing['erase_'+g]['early_epe']/continuing['moving']['early_epe']-1),preservation_criterion_met=a['candidates'][g]['passed'])
    partial_path=OUT/'partial_effects.json';partial_path.write_text(json.dumps(partials,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for g in partials['effects']:
        c['confirmed_results'].append(dict(id='partial_state_'+g+'_delay',title=LABELS[g]+' — задержка меньше, но полезный ответ ухудшен',category='representation',metric='flyvis_transition_delay_frames',unit='points',comparison='Исходная задержка минус задержка после подмены этой группы состоянием неподвижной сцены; один пункт метрики равен одному кадру.',subset='Четыре ранее изученных изображения BDD TRAIN, по четыре условия обычного контраста на каждое; группировка по исходной сцене.',conditions='Штатная модель flow/0000/000, одинаковый суффикс разворота, искусственный донор, CPU FP64; положительна также нижняя граница после поправки на шесть проверок.',limitations='Только частный диагностический эффект. Контроль продолжающегося движения ухудшен, общий критерий безопасной подмены не пройден. Это не готовое исправление, независимый итоговый тест или улучшение HOTA.',evidence=dict(source=str(partial_path.relative_to(ROOT)).replace('\\','/'),sha256=sha(partial_path),locator=['effects',g])))
    c['interpretation']+=' Отдельные сокращения задержки сохранены как частные диагностические сравнения вместе с неблагоприятными исходами; искусственный перенос не объявляется улучшением рабочего модуля.'
    cat.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    prob=ROOT/'experiments/system_problems.json';d=json.loads(prob.read_text(encoding='utf-8'));item=next(x for x in d['problems'] if x['id']=='С7')
    fact=c['obtained']+' '+interpretation+' '+limits+' '+decision+' '+literature
    item['cases'].append(dict(date=now.date().isoformat(),fact=fact,evidence=refs));item['cards'].append(c['id'])
    item['evidence'] += [dict(source=r,sha256=sha(ROOT/r),claim='Локализация предыстории с контролем продолжающегося движения') for r in refs]
    prob.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');journal(full+' Источник чисел: `'+refs[2]+'`.')
    with (ROOT/'FLYVIS_TRACKER_RESEARCH_ROADMAP.md').open('a',encoding='utf-8') as f:f.write('\n\n## '+now.strftime('%d.%m.%Y')+' — частичная подмена состояния\n\n'+fact+'\n')
    from scripts.obsidian_system_problems import load,problem_path
    from scripts.obsidian_fundamental_overview import build
    from scripts.audit_fundamental_obsidian import audit
    vault=ROOT/'ObsidianFlyVis';stage=OUT/'vault_checked';backup=OUT/'vault_before';assert not stage.exists() and not backup.exists()
    shutil.copytree(vault,stage);shutil.copytree(vault,backup);before={f.relative_to(vault).as_posix():sha(f) for f in vault.rglob('*') if f.is_file()}
    name=problem_path(next(x for x in load()['problems'] if x['id']=='С7'))+'.md';assert build(stage,[f for f in before if f.endswith('.md') and f!=name])==0
    for f in ('.obsidian/graph.json','.graph-visible.json','.generated.json'):shutil.copy2(backup/f,stage/f)
    check=audit(stage);assert not check['external_hypothesis_edges'];assert set(before)=={f.relative_to(stage).as_posix() for f in stage.rglob('*') if f.is_file()}
    changed=[f for f,h in before.items() if sha(stage/f)!=h];assert changed==[name] and all(sha(vault/f)==h for f,h in before.items())
    shutil.copy2(stage/name,vault/name);done=audit(vault);assert done['hashes']==check['hashes'];done.update(changed_files=changed,deleted_files=[],protected_hypotheses_byte_exact=True)
    (OUT/'graph_audit.json').write_text(json.dumps(done,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(intro)
if __name__=='__main__':main()
