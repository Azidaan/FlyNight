"""Independently reconstruct all learned action outcomes and record the complete bounded result."""
from pathlib import Path
from collections import Counter
import os,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
from scripts import z13_temporal_identity_trial_20261025 as S
from scripts.z13_temporal_identity_trial_recovery1_20261025 import retry_write,old_contrast
from scripts.z13_hota_deadline_guard import WindowsProcesses


def audit():
    p=S.read(S.RUN/'protocol.json');r=S.read(S.RUN/'report.json')
    assert not r.get('final_pending'), 'Final stage is still required after the internal pass'
    control_path=S.RUN/('final_execution/controller_result.json' if r.get('final_evaluated') and r.get('vectorized_execution') else 'recovery2/controller_result.json' if r.get('vectorized_execution') else 'controller_result.json')
    c=S.read(control_path);ps=WindowsProcesses()
    assert r['complete'] and c['exit_code']==0 and c['all_owned_ended'] and not c['remaining']
    assert all(ps.creation(int(pid))!=birth for pid,birth in c['owned'].items())
    assert S.read(S.RUN/'original_regression.json')['all_exact']
    entries={};uid_video={}
    for v in p['fit84']+p['fit20']+p['evaluate26']:
        for u,e in S.read(S.FOLLOW/'cache/videos'/(v+'.json'))['entries'].items():entries[int(u)]=e;uid_video[int(u)]=v
    data={};diagnostics={a:[] for a in S.ARMS};total_rows=0;models=0
    for name,digest in r['models_report_sha256'].items():
        folder=ROOT/name;rec=S.read(folder/'report.json');done=S.read(folder/'done.json')
        assert S.sha(folder/'report.json')==digest==done['report_sha256']
        assert S.sha(folder/'model.pt')==done['model_sha256'] and done['predictions_reproduced_exact'] and done['steps']==1150
        assert S.sha(folder/'rows.json')==rec['rows_sha256']
        rows=S.read(folder/'rows.json');parts=folder.relative_to(S.RUN).parts
        phase,arm,seed=parts[:3];seed=int(seed.removeprefix('seed'))
        fold=int(parts[3].removeprefix('fold')) if phase=='internal' else None
        expected=p['folds'][fold]['validation'] if phase=='internal' else p['evaluate26']
        per={v:[0,0] for v in expected};counts=Counter();seen=set()
        for row in rows:
            u=row['uid'];assert u not in seen;seen.add(u);e=entries[u];v=uid_video[u]
            assert row['video']==v and v in expected and row['kind']==e['kind']
            action=row['action'];assert -1<=action<len(e['choices']) and np.isfinite(row['probability'])
            known=e['kind']>=0;good=action>=0 and known and bool(e['good'][action]);harm=action>=0 and known and bool(e['harm'][action])
            assert row['beneficial']==good and row['harmful']==harm
            per[v][0]+=int(good)-int(harm);per[v][1]+=known
            counts.update(dict(rows=1,selected=int(action>=0),beneficial=int(good),harmful=int(harm),unknown_selected=int(action>=0 and not known)))
        values={v:a/b for v,(a,b) in per.items()};assert values==rec['per_video']
        assert all(counts[k]==v for k,v in rec['counts'].items())
        key=(phase,arm,seed);saved=data.setdefault(key,{})
        assert not set(saved)&set(values);saved.update(values)
        if phase=='internal':diagnostics[arm].append(rec['identity_diagnostic'])
        models+=1;total_rows+=len(rows)
    assert models==r['models'] and models in (90,108)
    videos=p['fit84']+p['fit20'];means={a:{} for a in S.ARMS}
    for a in S.ARMS:
        for seed in p['seeds']:assert set(data['internal',a,seed])==set(videos)
        means[a]={v:float(np.mean([data['internal',a,s][v] for s in p['seeds']])) for v in videos}
    internal=S.read(S.RUN/'internal_report.json');comparison=S.contrast(means)
    assert comparison==internal['paired'] and means==internal['means'] and S.passes(comparison)==internal['admit_final']==r['internal_pass']
    final=None
    if r['final_evaluated']:
        fm={a:{v:float(np.mean([data['final',a,s][v] for s in p['seeds']])) for v in p['evaluate26']} for a in S.ARMS}
        fc=S.contrast(fm);fc['D-original_context']=old_contrast(fm['D'],S.read(S.FOLLOW/'training/report.json')['mean_seed_per_video']['D_context'])
        final=S.read(S.RUN/'final_report.json');assert fc==final['paired'] and fm==final['means'] and S.passes(fc)==final['primary_pass']==r['final_pass']
    oracle={}
    for subset,vs in [('fit104',videos),('evaluate26',p['evaluate26'])]:
        values={}
        for v in vs:
            known=[e for u,e in entries.items() if uid_video[u]==v and e['kind']>=0]
            values[v]=sum(any(e['good']) for e in known)/len(known)
        oracle[subset]=dict(mean_per_video=float(np.mean(list(values.values()))),per_video=values,
            includes_explicit_null=True,warning='Ideal selection over exact learned action set; not predicted performance. Does not isolate incremental temporal benefit.')
    diag={}
    for a,records in diagnostics.items():
        auc=[v['candidate_identity_auroc'] for v in records if v['candidate_identity_auroc'] is not None]
        diag[a]=dict(model_reports=len(records),mean_fold_seed_auroc=float(np.mean(auc)) if auc else None,
            known_pairs=sum(v['known_pairs'] for v in records),positives=sum(v['positives'] for v in records),
            caveat='Descriptive mean of fold/seed AUCs,not an independent pair/video confidence interval or HOTA.')
    receipt=dict(at=S.now().isoformat(),models=models,action_rows_verified=total_rows,all_model_and_row_hashes_verified=True,
        all_saved_actions_rebound_to_original_outcomes=True,all_primary_means_and_intervals_recomputed=True,
        internal_gate_recomputed=internal['admit_final'],final_gate_recomputed=final['primary_pass'] if final else None,
        matched_action_oracle=oracle,identity_diagnostics=diag,all_owned_ended=True,
        successful_controller_source=control_path.relative_to(ROOT).as_posix(),
        original_failed_price_controller_preserved=bool(r.get('vectorized_execution')))
    retry_write(S.RUN/'independent_audit.json',receipt)
    return p,r,internal,final,receipt


def record():
    assert not (S.RUN/'recorded.json').exists()
    p,r,internal,final,audit_result=audit();passed=bool(r['final_pass']) if r['final_evaluated'] else False
    source=S.RUN/('final_report.json' if final else 'internal_report.json');result=final or internal
    cp=ROOT/'experiments/project_history.json';catalog=S.read(cp);card=next(c for c in catalog['experiments'] if c['id']==S.IDENT)
    count=r['models'];subset='26повторноиспользованныхвидео' if final else '104видео вложеннойпроверки'
    summary=('Критерии этой реализации пройдены; нужен отдельный стендовый опыт.' if passed else 'Межкадровый считыватель с метками личности не прошёл критерии этой реализации.')
    obtained=source.relative_to(ROOT).as_posix()+' → paired: '+str(result['paired'])+'; '+(S.RUN/'report.json').relative_to(ROOT).as_posix()+':models='+str(count)+'. independent_audit.json:всеисходы,веса,сохранённыеповторыпредсказанийиинтервалыпроверены.'
    interpretation=('Проверено внутреннее считывание реальных областей,одинаковаяопорнаясетка иявноеразмеченноесопоставление прификсированномядре. '+
        ('Получен допуск этой реализации к отдельной проверке полного стенда.' if passed else 'Одна эта реализация не реализовала заранее заданную чистую пользу и полный набор преимуществ над контролями.')+
        ' Это '+subset+'; условные интервалы и одноступенчатые назначения,неHOTA. Диагностика соответствий сама по себе не доказывает пользу вмешательства или причину неудачи.')
    decision=('Сохранить модель как кандидат; подготовить отдельно согласованный замкнутый опыт с исходным стендом,C,R1,контролембезмухи. HOTA этой очередью не запускается.' if passed else
        'Эту реализацию не переносить в стенд,не подбирать порог/сид/длину по её исходу. Сохранить узел и полный отрицательный результат. Следующее изменение выбирать по различию обучения соответствий и полезного выбора,не объявляя линиюFlyVis закрытой.')
    if not final:decision+=' Новая26оценка не запускалась после внутренней остановки.'
    card.update(state='completed_positive' if passed else 'completed_negative',done=S.now().strftime('%Y-%m-%d'),summary=summary,
        obtained=obtained,interpretation=interpretation,decision=decision,confirmed_results=[])
    card['acceptance'][0].update(met=True,evidence='runs/z13_temporal_identity_20261025/training/original_regression.json; independent_audit.json')
    card['acceptance'][1].update(met=True,evidence='runs/z13_temporal_identity_20261025/features/independent_audit.json;training/independent_audit.json')
    card['acceptance'][2].update(met=internal['admit_final'],evidence='runs/z13_temporal_identity_20261025/training/internal_report.json')
    card['acceptance'][3].update(met=final['primary_pass'] if final else None,evidence='runs/z13_temporal_identity_20261025/training/'+('final_report.json' if final else 'internal_report.json; final stage not run'))
    card['evidence'] += [dict(kind='file',ref=(S.RUN/name).relative_to(ROOT).as_posix(),claim=claim) for name,claim in [
        ('independent_audit.json','Независимый полный пересчёт исходов и интервалов'),('report.json','Число моделей,ступени,отсутствиеHOTA'),
        ('original_regression.json','Точное воспроизведение12исходныхголов')]]
    card['evidence'].append(dict(kind='file',ref=audit_result['successful_controller_source'],claim='Успешный контроллер и полное завершение дерева; прежний неудачный замер цены сохранён отдельно'))
    gp=ROOT/'experiments/metrics_glossary.json';glossary=S.read(gp)
    if 'temporal_readout_net_assignment_effect' not in glossary['metrics']:
        glossary['metrics']['temporal_readout_net_assignment_effect']=dict(title='Чистая польза выбора рамочного действия',
            ru='Средняя по видео разность числа полезных и вредных выбранных действий,делённая на все известные исходные запросы каждого видео.',
            better='higher',unit='fraction',range=[-1,1],tool='Точные сохранённые одноступенчатые исходы назначения',
            caveat='Не HOTA; неизвестные случаи отдельно,все известные остаются в знаменателе. Для парного сравнения фиксируются действия,состояния,видео,сид иразбиение.')
        retry_write(gp,glossary)
    card['metrics']=[dict(key='temporal_readout_net_assignment_effect',value=result['paired']['D-baseline']['mean_per_video'],baseline=0,
        subset=subset,source=source.relative_to(ROOT).as_posix(),locator='paired.D-baseline.mean_per_video')]
    confirmed=[]
    for key,value in result['paired'].items():
        if value['ci95'][0] <= 0:
            continue
        comparison=key.replace('D-baseline','D против неизменного назначения').replace('D-D_blank','D с историей против той же модели без истории').replace('D-D_no_identity','D с метками личности против той же истории без этих меток')
        confirmed.append(dict(id='temporal-identity-'+key.lower().replace('_','-'),
            title='Частный эффект выбора действий: '+comparison,category='representation' if key in ('D-R1','D-geometry','D-C') else 'readout',
            metric='temporal_readout_net_assignment_effect',unit='fraction',comparison=comparison,subset=subset,
            conditions='Фиксированные исходные состояния и действия; одинаковые признаки/узел/бюджет,три сида и видео-раздельная калибровка; изменяется условие указанного сравнения.',
            limitations='Частный контраст не меняет общий исход карточки и не отменяет непройденные критерии. Повторные видео,условные интервалы;неHOTA,не независимое подтверждение всей модели. Ядро в этом опыте не обучалось.',
            evidence=dict(source=source.relative_to(ROOT).as_posix(),sha256=S.sha(source),locator=['paired',key])))
    card['confirmed_results']=confirmed
    retry_write(S.RUN/'confirmed_results_review.json',dict(at=S.now().isoformat(),source_sha256=S.sha(source),
        reviewed_all_preregistered_contrasts=True,new_confirmed_results=len(confirmed),primary_pass=passed,
        pair_auroc_excluded_because_no_video_interval=True))
    retry_write(cp,catalog)
    S.note('итог опыта межкадрового считывания личности','*Получено (`'+S.IDENT+'`).* '+obtained+'\n\n*Вывод.* '+interpretation+'\n\n*Решение.* '+decision+
        '\n\n*Доказательства.* `runs/z13_temporal_identity_20261025/training/independent_audit.json`; все обученные модели и неблагоприятные исходы сохранены. '+
        '\n\n*Подборка подтверждённых результатов.* Проверены все заданные контрасты; добавлено'+str(len(confirmed))+'частных сравнений с положительной нижнейграницей и неизменным первичнымJSON. Общий исход карточки и непройденные критерии сохранены; улучшениеHOTA не измерялось. '+
        '\n\n*Сводка к исходному стенду.* HOTA не измерялась. Доля реализованного запаса одноступенчатых действий='+str(result['paired']['D-baseline']['mean_per_video']/audit_result['matched_action_oracle']['evaluate26' if final else 'fit104']['mean_per_video'])+
        '; её знаменатель включаетNULL и совпадает с действиями модели. Это не процент улучшенияHOTA.')
    retry_write(S.RUN/'recorded.json',dict(at=S.now().isoformat(),complete=True,primary_pass=passed,
        sentence_review=dict(method_distinguished_from_object=True,no_duplicate_scientific_claim=True,only_completed_work=True)))


if __name__=='__main__':
    if '--record' in sys.argv:record()
    elif '--audit' in sys.argv:print(audit()[-1])
