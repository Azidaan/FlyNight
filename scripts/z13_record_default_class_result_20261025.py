"""Independent primary-row audit and final record; never retrains models."""
from pathlib import Path
import subprocess,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts.z13_margin_rule_audit_20261025 import revise
from scripts.z13_default_class_trial_20261025 import journal,comparisons
from scripts.z13_localization_labels_20261023 import verify_sources
from scripts.z13_hota_deadline_guard import WindowsProcesses
RUN=ROOT/'runs/z13_default_class_trial_20261025'
OLD=ROOT/'runs/z13_box_supervision_followup_20261025'


def record():
    assert not (RUN/'finish_recorded.json').exists()
    p=read(RUN/'protocol.json');verify_sources(p);r=read(RUN/'report.json');internal=read(RUN/'internal_report.json')
    assert r['complete'] and r['models']==120 and not r['internal_pass'] and not r['final_evaluated']
    assert sha(RUN/'internal_report.json')==r['internal_report_sha256'] and not (RUN/'final_report.json').exists()
    end=read(RUN/'controller_result.json');ps=WindowsProcesses()
    assert end['exit_code']==0 and end['worker_ended'] and end['all_owned_ended'] and not end['remaining']
    assert all(ps.creation(int(pid))!=b for pid,b in end['owned'].items())
    videos=p['fit84']+p['fit20'];entries={};video_of={}
    for v in videos:
        path=OLD/'cache/videos'/(v+'.json');assert sha(path)==read(OLD/'cache/report.json')['manifest'][v]
        for u,x in read(path)['entries'].items():
            u=int(u);assert u not in entries;entries[u]=x;video_of[u]=v
    aggregate={};models=0;source_hashes={}
    for arm in p['arms']:
        for loss in p['losses']:
            for seed in p['seeds']:
                per={rule:{} for rule in p['rules']};uids={rule:set() for rule in p['rules']}
                for f,split in enumerate(p['folds']):
                    folder=RUN/'internal'/arm/loss/f'seed{seed}'/f'fold{f}'
                    done=read(folder/'done.json');rec=read(folder/'report.json')
                    assert done['steps']==p['steps'] and done['predictions_reproduced_exact']
                    assert sha(folder/'model.pt')==done['model_sha256'] and sha(folder/'report.json')==done['report_sha256'];models+=1
                    source_hashes[(folder/'model.pt').relative_to(ROOT).as_posix()]=done['model_sha256']
                    for rule in p['rules']:
                        rows=read(folder/rule/'rows.json');assert sha(folder/rule/'rows.json')==rec[rule]['rows_sha256']
                        source_hashes[(folder/rule/'rows.json').relative_to(ROOT).as_posix()]=rec[rule]['rows_sha256']
                        expected={u for u,v in video_of.items() if v in split['validation']}
                        assert {x['uid'] for x in rows}==expected and len(rows)==len(expected) and not uids[rule]&expected
                        uids[rule]|=expected;counts=dict(selected=0,beneficial=0,harmful=0,unknown_selected=0)
                        numerator={v:0 for v in split['validation']};denominator=dict(numerator)
                        for x in rows:
                            en=entries[x['uid']];a=x['action'];k=x['kind']>=0
                            assert x['video']==video_of[x['uid']] and x['kind']==en['kind'] and -1<=a<len(en['choices'])
                            good=bool(k and a>=0 and en['good'][a]);harm=bool(k and a>=0 and en['harm'][a])
                            assert good==x['beneficial'] and harm==x['harmful']
                            counts['selected']+=a>=0;counts['beneficial']+=good;counts['harmful']+=harm;counts['unknown_selected']+=a>=0 and not k
                            denominator[x['video']]+=k;numerator[x['video']]+=int(good)-int(harm)
                        actual={v:numerator[v]/denominator[v] for v in numerator}
                        assert actual==rec[rule]['per_video'] and counts==rec[rule]['counts'];per[rule].update(actual)
                        grid=rec[rule]['calibration'];assert len(grid)==11
                        for g in grid:
                            assert set(g['per_video'])==set(split['calibration'])
                            assert g['mean']==float(np.mean(list(g['per_video'].values())))
                        best=max(range(len(grid)),key=lambda j:(grid[j]['mean'],-grid[j]['counts']['selected'],j))
                        assert grid[best]['threshold']==rec[rule]['threshold']
                assert all(u==set(entries) for u in uids.values())
                aggregate[arm,loss,seed]=per
    means={loss:{rule:{arm:{v:float(np.mean([aggregate[arm,loss,s][rule][v] for s in p['seeds']])) for v in videos} for arm in p['arms']} for rule in p['rules']} for loss in p['losses']}
    comp=comparisons(means);assert means==internal['means'] and comp==internal['paired'] and models==120
    passed=comp['D-baseline']['mean_per_video']>=.02 and all(c['ci95'][0]>0 for c in comp.values());assert passed==internal['admit_final']==False
    write(RUN/'independent_audit.json',dict(at=now().isoformat(),models=models,policies=240,all_primary_rows_bindings_labels_counts_means_and_intervals_verified=True,all_threshold_selections_match_calibration=True,all_owned_ended=True,source_hashes=source_hashes,report_sha256=sha(RUN/'internal_report.json')))
    summary={}
    for loss in p['losses']:
        summary[loss]={}
        for rule in p['rules']:
            summary[loss][rule]={a:dict(mean_net=float(np.mean(list(means[loss][rule][a].values())))) for a in p['arms']}
    write(RUN/'final_summary.json',dict(complete=True,internal_pass=False,models=120,paired=comp,all_cells=summary,seconds=r['seconds'],train26_evaluated=False,hota_measured=False,confirmed_results_eligible=0))
    cat=read(ROOT/'experiments/project_history.json');card=next(x for x in cat['experiments'] if x['id']==p['id'])
    reason='Завершены120внутренних моделей; независимая проверка первичных строк, хешей, разбиений и интервалов.'
    values=dict(summary='Обучаемый отказ и стоимостная функция ошибки: внутренний допуск не пройден;120моделей наfit104. Новых оценок26/HOTA нет.',
        obtained='runs/z13_default_class_trial_20261025/internal_report.json → paired: '+str(comp)+';120моделей,5внешних частей,3сида;1694,718сCPU. Проверка: independent_audit.json в том же каталоге.',
        interpretation='Новый внутренний считыватель с явным отказом и costCE не дал подтверждённой чистой пользы или прибавки к MSE/C/R1/геометрии. Это отрицательный итог данного устройства и бюджета, не доказательство отсутствия информации в активности FlyVis. Интервалы условны на фиксированном вложенном разбиении повторныхfit104; не являются оценкой нового набора видео или HOTA.',
        decision='Остановить эту реализацию по предварительному внутреннему допуску; не обучать24итоговые головы, не оценивать26 и не запускатьHOTA. Не перебиратьepsilon/пороги на том же результате. Следующий кандидат — пространственное считывание по неусреднённой карте; сначала проверить отличие от прежних пространственных проб и доступность исходных карт. Новая модель пока не подготовлена.',
        cannot_claim='Нельзя заявлять исправленный отказ, преимущество D, нулевой истинный эффект, отсутствие информации по провалу моделей, улучшениеHOTA или окончательный отрицательный вывод обо всей FlyVis. С2 не проводилась; новыйtrain26результат отсутствует. Теоретическая согласованность потерь не гарантировала пользу при данных признаках/модели.')
    for k,v in values.items():revise(card,k,v,reason)
    card['state']='completed_negative';card['done']=now().strftime('%Y-%m-%d')
    card['acceptance'][2].update(met=False,evidence='runs/z13_default_class_trial_20261025/internal_report.json: D-baseline и D-geometry интервалы включают0;дополнительно mean<.02')
    for i in (3,4):card['acceptance'][i].update(met=None,evidence='С2 не запускалась после непройденного внутреннего допуска; результат26неизмерен')
    card['acceptance'][5].update(met=True,evidence='runs/z13_default_class_trial_20261025/independent_audit.json; все3сида, фиксированнаясетка,0новыхоценок26/HOTA')
    card['evidence'].append(dict(kind='json',ref='runs/z13_default_class_trial_20261025/internal_report.json',claim='Полный отрицательный внутренний итог; все основные сравнения'))
    write(ROOT/'experiments/project_history.json',cat);write(RUN/'finish_recorded.json',dict(at=now().isoformat(),complete=True,internal_pass=False,final_evaluated=False))
    write(RUN/'confirmed_review.json',dict(eligible_new_confirmed_results=0,reason='All five primary paired intervals include zero.'))
    journal('итог120моделей обучаемого отказа','*Получено.* '+values['obtained']+'\n\n*Вывод.* '+values['interpretation']+'\n\n*Решение.* '+values['decision']+'\n\n*Архитектурная веха.* Добавить проверенный узел явного отказа и обучения стоимости; жёлтый, отрицательный внутренний результатfit104, неtrain26. Одинлист;непересчитыватьнаучныеисточники.\n\nHOTA/проценткстенду/доляпотолканеопределены.')
    h=ROOT/'runs/z13_weekend_20261024/HANDOFF.md'
    h.write_text('# Актуально — '+now().strftime('%d.%m.%Y %H:%M')+' MSK\n\nЗавершён z13_default_class_readout_v1:120моделей,exit0,своёдеревозавершено,1694,718с. D-baseline+0,01647п.п.[-0,05855;+0,10791];все5ДИвключают0,допускнепройден.26иHOTAнезапускались. independent_audit/final_summary/finish_recorded в runs/z13_default_class_trial_20261025. Следующаякандидатура—неусреднённоепространственноесчитывание;сначалааудитдоступностикартиотличияотmaxpool/производных/4bins. Старые4bins:inconclusive,порогнельзяослаблять. Готовогоисполнителянет,невыдаватьпланзапрогон. Нужно завершитьархитектуру/тесты/граф/точныйpushэтогоитога. Общийсрок26.10до07:40МСК;всёскрыто,57чужихправок/.gitignoreсохранить.\n\nПрежние передачи сохранены.\n\n'+h.read_text(encoding='utf8'),encoding='utf8')


if __name__=='__main__':record()
