"""Register execution and publish only this bounded pilot's own sources and results."""
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.z13_parallel_next import read,write,sha
from scripts.z13_pairwise_utility_20261021 import RUN,IDENT,verify,journal


def launch_record():
    import psutil
    launch=read(RUN/'launch.json');control=read(RUN/'controller.json')
    process=psutil.Process(launch['observer_launcher'])
    assert 'z13_pairwise_utility_progress.ps1' in ' '.join(process.cmdline())
    assert launch['observer_window_handle']>0
    path=ROOT/'experiments/project_history.json';catalog=read(path)
    card=next(x for x in catalog['experiments'] if x['id']==IDENT)
    assert card['state']=='ready'
    card['state']='running'
    path.write_text(json.dumps(catalog,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    journal('запуск ограниченной проверки', '*Получено.* runs/z13_pairwise_utility_20261021/launch.json: проверено отдельное видимое окно PowerShell с названием опыта; controller.json: независимый контроллер запущен '+control['started_at']+', сохранение до '+control['save_stop_at']+', полное завершение до '+control['hard_stop_at']+'.\n\n*Решение.* Выполнить заранее заданную проверку без изменений критериев, данных и ядра. Опрос чата не создаётся; консоль читает состояние локально.')
    write(RUN/'execution_review.json',dict(recorded_at=datetime.now(timezone(timedelta(hours=3))).isoformat(),
          observer_verified=True,controller_protocol=control,service_source_sha256=sha(Path(__file__))))


def publish():
    os.environ['GIT_OPTIONAL_LOCKS']='0'
    report=read(RUN/'report.json');end=read(RUN/'controller_result.json')
    assert end['worker_ended'] and end['exit_code']==0 and not end['hard_timeout']
    assert report['reproduction_pass'] and report['source_hashes_unchanged']
    assert read(RUN/'result_audit.json')['passed']
    verify(read(RUN/'protocol.json'))
    assert read(RUN/'validation.json')['accepted']
    preservation=read(RUN/'vault_preservation.json')
    protected={p.replace(chr(92),'/') for p in preservation['protected_existing_edits']}
    protected.update(p.replace(chr(92),'/') for p in read(RUN/'vault_refresh.json')['protected_existing_edits'])
    own=['tracking/z13_pairwise_utility.py','scripts/z13_pairwise_utility_20261021.py',
         'scripts/z13_pairwise_utility_progress.ps1','scripts/z13_publish_pairwise_20261021.py',
         'tests/test_z13_pairwise_utility.py','experiments/project_history.json',
         'dissertation/ЕДИНЫЙ_ПЛАН.md','ObsidianFlyVis/01 Муха/Эксперименты/'+IDENT+'.md']
    records=['protocol.json','reproduction.json','report.json','controller_result.json','result_audit.json']
    for arm in ('D','C','R1'):
        records += [arm+'/'+name for name in ('done.json','scores.json','folds.json')]
    paths=sorted(set(own+[(RUN/p).relative_to(ROOT).as_posix() for p in records]))
    assert not set(paths)&protected
    assert all((ROOT/p).is_file() for p in paths)
    write(RUN/'commit_scope.json',dict(paths=paths,excluded='Foreign edits, unrelated graph files, run logs, caches and binary models'))
    def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT).decode('utf-8').strip()
    assert git('remote','get-url','origin')=='https://gitlab.com/aziteam-group/drbdiss.git'
    assert git('branch','--show-current')=='main'
    assert not (ROOT/'.git/index.lock').exists()
    staged=set(filter(None,git('diff','--cached','--name-only','-z').split('\0')))
    assert staged<=set(paths),'Unrelated staged files'
    subprocess.run(['git','add','-f','--',*paths],cwd=ROOT,check=True)
    assert set(filter(None,git('diff','--cached','--name-only','-z').split('\0')))<=set(paths)
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
    subprocess.run(['git','commit','-m','Test a fixed pairwise utility objective on saved FlyVis features',
                    '-m','Co-authored-by: Codex <codex@openai.com>'],cwd=ROOT,check=True)
    head=git('rev-parse','HEAD')
    result=subprocess.run(['git','push','origin','main'],cwd=ROOT,capture_output=True,text=True,timeout=90)
    write(RUN/'publish_result.json',dict(commit=head,published=result.returncode==0,exit_code=result.returncode,message=(result.stdout+result.stderr)[-3000:]))
    print(json.dumps(dict(commit=head,published=result.returncode==0)))
    assert result.returncode==0


def audit():
    import numpy as np
    from scripts.z13_continue import paired,per_video_useful,coverage_table
    from scripts.z13_score_swap import key,selected_keys
    p=read(RUN/'protocol.json');verify(p)
    report=read(RUN/'report.json');end=read(RUN/'controller_result.json')
    assert end['worker_ended'] and end['exit_code']==0 and not end['hard_timeout']
    levels={};counts={}
    for arm in p['arms']:
        rows=read(ROOT/f'runs/z13_target_center_20261018/{arm}/center_rows.json')
        predictions=read(RUN/arm/'scores.json')
        assert set(predictions)=={'point_map','pair_map','point_geometry','pair_geometry'}
        variants={}
        for name,items in predictions.items():
            values={key(dict(x,label=str(x['label']))):x['score'] for x in items}
            assert len(values)==len(rows) and set(values)=={key(r) for r in rows}
            assert all(np.isfinite(x) and 0<=x<=1 for x in values.values())
            variants[name]=[dict(r,p_present=values[key(r)]) for r in rows]
        old=read(ROOT/f'runs/z13_map_utility_20261018/{arm}/predicted_scores.json')
        score_by_id={(x['video'],str(x['label'])):x['score'] for x in old}
        variants['old_map']=[dict(r,p_present=score_by_id[r['video'],str(r['label'])]) for r in rows]
        for name,scored in variants.items():
            counts[arm,name]=coverage_table(scored)
            assert counts[arm,name]==report['arms'][arm]['coverage'][name]
            for q in (.01,.02,.05):
                selected=selected_keys(scored,q)
                pv=per_video_useful([dict(r,useful=bool(r['useful']) and key(r) in selected) for r in scored if r['kind']==1])
                levels[arm,name,q]=pv
                actual=float(np.mean(list(pv.values())))
                assert actual==report['arms'][arm]['levels'][f'{name}/{q:.2f}']
    primary={f'pair_map-{b}':paired(levels['D','pair_map',.01],levels['D',b,.01]) for b in ('point_map','old_map','pair_geometry')}
    primary['D_pair_map-R1_pair_map']=paired(levels['D','pair_map',.01],levels['R1','pair_map',.01])
    assert primary==report['primary']
    assert report['primary_pass']==all(x['ci95'][0]>0 for x in primary.values())
    before=report['arms']['D']['levels']['old_map/0.01']
    after=report['arms']['D']['levels']['pair_map/0.01']
    write(RUN/'result_audit.json',dict(passed=True,all_saved_predictions_checked=True,
          primary_values_recomputed_from_scores=True,all_coverage_and_levels_match=True,
          original_sources_unchanged=True,unknowns_imputed=False,
          baseline_mean_per_video=before,new_mean_per_video=after,
          relative_percent=100*(after/before-1),ceiling_fraction=None))
    journal('проверка первичного результата и границ вывода',
        '*Получено.* runs/z13_pairwise_utility_20261021/result_audit.json: все сохранённые ключи, оценки и доли отбора сверены; основные интервалы независимо пересчитаны по scores.json и точно совпали с report.json. Прежние источники неизменны.\n\n*Вывод.* При отборе1% доля положительных случаев с полезной выбранной поправкой D в среднем по видео '+
        f'{100*before:.4f}% → {100*after:.4f}%; относительное изменение {100*(after/before-1):+.4f}%'.replace('.',',')+
        '; интервал парной добавки содержит ноль. Доля потолка не определена — соответствующий потолок не измерялся. HOTA не измерялась.\n\n*Решение.* Новых сравнений для раздела лучших результатов не добавлено: основной критерий не пройден; вторичные2%/5% сохранены в полном отчёте и не заменяют его. Статус проблемы P8 не меняется. Это проверка способа применения сигнала, а не новая модернизация ядра FlyVis.')


def localize_and_refresh():
    """Translate display text and regenerate our newly created card only."""
    from scripts import z13_build_vault_preserving_edits as wrapper
    card_path=ROOT/'ObsidianFlyVis/01 Муха/Эксперименты'/ (IDENT+'.md')
    baseline=read(RUN/'vault_preservation.json')['protected_existing_edits']
    rel=card_path.relative_to(ROOT).as_posix()
    assert rel not in {x.replace(chr(92),'/') for x in baseline}
    digest=sha(card_path)
    path=ROOT/'experiments/project_history.json';catalog=read(path)
    card=next(x for x in catalog['experiments'] if x['id']==IDENT)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
    fields=dict(question='Улучшит ли добавление парной ошибки отбор полезных поправок при неизменных признаках и основном отборе1%?',
        decision='Критерий фиксированной парной цели не пройден; эту реализацию не переносить в трекер и не подбирать по dev.',
        cannot_claim='Повторно использованный train26, одно фиксированное начальное состояние генератора случайных чисел. Неизвестные исходы исключены из обучения, но занимают места при отборе. Проверялся способ применения модернизированной FlyVis, а не новая модернизация ядра. Не измерялись HOTA, перенос на dev или отложенные75 и полный риск неизвестных исходов. Дополнительные доли2%/5% не заменяют основной критерий; известная неполезная поправка не равна измеренному вреду трекеру.')
    for field,value in fields.items():
        if card.get(field)!=value:
            card.setdefault('revisions',[]).append(dict(date='2026-10-21',commit=head,field=field,old=card.get(field,''),
                reason='Перевод отображаемого текста до первого коммита этой карточки; hash указывает базовый коммит рабочего дерева, прежний текст сохранён здесь. Протокол, критерии и числа не менялись.'))
        card[field]=value
    path.write_text(json.dumps(catalog,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    journal('уточнение языка карточки', 'Вопрос, решение и границы вывода записаны по-русски; прежние формулировки сохранены в revisions. Числа, протокол и критерии неизменны. Предмет — FlyVis; оценка отбора — способ применения сигнала; результат относится только к выполненной проверке по сохранённым train26, без HOTA. Текст проверен на различие предмета и инструмента, повторение соседних полей и соответствие фактически выполненному.')
    original=wrapper.snapshot_local_edits
    def owned_snapshot(root):
        protected=original(root)
        assert protected[card_path.resolve()]==digest, 'Our generated card changed after ownership check'
        del protected[card_path.resolve()]
        return protected
    wrapper.snapshot_local_edits=owned_snapshot
    argv=sys.argv
    try:
        sys.argv=['z13_build_vault_preserving_edits.py','--report',str(RUN/'vault_refresh.json')]
        wrapper.main()
    finally:
        sys.argv=argv;wrapper.snapshot_local_edits=original
    write(RUN/'owned_card_refresh.json',dict(path=rel,previous_sha256=digest,new_sha256=sha(card_path),
          absent_from_original_foreign_edits=True,official_generator_used=True))


if __name__=='__main__':
    {'launch':launch_record,'publish':publish,'audit':audit,'localize':localize_and_refresh}[sys.argv[1]]()
