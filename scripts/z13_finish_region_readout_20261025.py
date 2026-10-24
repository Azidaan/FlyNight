"""Local completion queue: wait without model calls, verify, record and publish."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timedelta
import inspect
import json
import os
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.z13_region_readout_20261025 import RUN,W,IDENT,ARMS,ALPHAS,read,write,sha,now,journal
ORIGINAL_RUN=RUN
if '--recovery1' in sys.argv:
    RUN=RUN/'recovery1'


def record():
    from scripts.z13_hota_deadline_guard import WindowsProcesses
    from scripts.z13_localization_labels_20261023 import verify_sources
    from scripts.z13_continue import paired
    assert not (RUN/'finish_recorded.json').exists()
    p,e=read(RUN/'protocol.json'),read(RUN/'controller_result.json')
    assert e['worker_ended'] and e['all_owned_ended'] and not e['remaining']
    processes=WindowsProcesses()
    assert all(processes.creation(int(pid))!=birth for pid,birth in e['owned'].items())
    verify_sources(p)
    complete=e['exit_code']==0 and (RUN/'report.json').exists()
    source=(RUN/'report.json').relative_to(ROOT).as_posix()
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
    if complete:
        r=read(RUN/'report.json');assert r['complete'] and r['exact_baseline_videos']==46
        choice=read(RUN/'choice.json');assert choice['frozen_before_evaluate26']
        all_values={a:{j:{} for j in range(4)} for a in ARMS}
        counts={a:[Counter() for _ in ALPHAS] for a in ARMS}
        for video in p['videos']:
            path=RUN/'videos'/(video+'.json');assert sha(path)==r['video_sha256'][video]
            job=read(path);assert job['checks']['raw_replay_exact'] and job['checks']['all_predictions_and_identities_bound']
            known=[x for x in job['rows'] if x['kind']>=0];assert known
            for a in ARMS:
                for j in range(4):
                    all_values[a][j][video]=float(sum(int(x['arms'][a][j]['beneficial'])-int(x['arms'][a][j]['harmful']) for x in known)/len(known))
                    if video not in p['evaluate26']:continue
                    for x in job['rows']:
                        c=counts[a][j];c['rows']+=1
                        if x['kind']<0:c['unknown']+=1
                        else:
                            c['known']+=1;c['beneficial']+=x['arms'][a][j]['beneficial'];c['harmful']+=x['arms'][a][j]['harmful']
        indices={a:max(range(4),key=lambda j:float(np_mean([all_values[a][j][v] for v in p['fit20']]))) for a in ARMS}
        assert indices==choice['indices']
        pv={a:{v:all_values[a][indices[a]][v] for v in p['evaluate26']} for a in ARMS}
        assert pv==r['per_video']
        assert r['counts']=={a:dict(counts[a][indices[a]]) for a in ARMS}
        comp={'D-baseline':paired(pv['D'],{v:0. for v in p['evaluate26']})}
        for a in ('C','R1','geometry'):comp['D-'+a]=paired(pv['D'],pv[a])
        assert comp==r['paired']
        passed=comp['D-baseline']['mean_per_video']>=.02 and all(x['ci95'][0]>0 for x in comp.values())
        assert passed==r['primary_pass']
        fields=dict(summary='Проверен конкретный региональный декодер; основной критерий '+('пройден.' if passed else 'не пройден.'),
            obtained=source+' → choices: '+json.dumps(r['choices'])+'; counts: '+json.dumps(r['counts'])+'; paired: '+json.dumps(r['paired'])+'. Полные46 исходных выходов повторены точно; reproduction.json и videos/*.json проверены.',
            interpretation='Здесь менялся выход внутреннего считывания: оценка реальных областей вместо переноса рамки на один пик. '+
                ('Получен предварительный сигнал сверх одинаковых контролей. Он относится к отдельным назначениям на неизменных состояниях, не к полному трекеру.' if passed else
                 'Эта конкретная версия не подтверждает преимущество D сверх всех обязательных контролей и практического порога. Неудача не устанавливает невозможность другого регионального считывания.'),
            decision=('Подготовить отдельную проверку стоимости и замороженного применения с теми же контролями; не объявлять прирост HOTA до замкнутого прогона.' if passed else
                      'Не внедрять этот декодер в стенд и не подбирать alpha по26видео. Сохранить реализацию и полную сетку калибровки, включая нулевой вариант.'),
            cannot_claim=p['counterargument']+' '+p['limits'])
        state='completed_positive' if passed else 'inconclusive'
    else:
        passed=None
        count=len(list((RUN/'videos').glob('*.json')))
        fields=dict(summary='Сохранён неполный служебный результат регионального декодера.',
            obtained=(RUN/'controller_result.json').relative_to(ROOT).as_posix()+': exit='+str(e['exit_code'])+'; готово '+str(count)+'/46 видео. Исходный код ошибки и журналы сохранены.',
            interpretation='Неполный проход не даёт вывода об эффективности этой версии.',
            decision='Не продвигать в HOTA. Проверить служебную причину и готовые файлы; исправление допустимо только в прежнем окне, без повторения готового.',cannot_claim=p['limits'])
        state='inconclusive'
    cp=ROOT/'experiments/project_history.json';cat=read(cp);card=next(c for c in cat['experiments'] if c['id']==IDENT)
    for key,value in fields.items():
        if card.get(key) and card[key]!=value:
            card.setdefault('revisions',[]).append(dict(date=now().strftime('%Y-%m-%d'),commit=head,field=key,old=card[key],reason='Сохранён проверенный итог конкретного регионального декодера.'))
        card[key]=value
    card.update(state=state,done=now().strftime('%Y-%m-%d'))
    card['acceptance'][0].update(met=True if complete else None,evidence=(RUN/'finish_recorded.json').relative_to(ROOT).as_posix())
    card['acceptance'][1].update(met=passed,evidence=source if complete else 'Неполный проход')
    if complete:card['evidence'].append(dict(kind='file',ref=source,claim='Все назначения, одинаковые контроли и парные интервалы'))
    write(cp,cat)
    journal('итог регионального выхода считывателя',
        '*Получено.* '+fields['obtained']+'\n\n*Вывод.* '+fields['interpretation']+'\n\n*Решение.* '+fields['decision']+
        '\n\nHOTA не измерялась; процент к стенду и доля достижимого потолка не определены. После завершения блока проверить архитектурную веху: новый вход областей и выход оценок должен отображаться отдельной ветвью, не улучшением всей мухи.')
    write(RUN/'finish_recorded.json',dict(at=now().isoformat(),complete=complete,primary_pass=passed,all_owned_ended=True,
        report_sha256=sha(RUN/'report.json') if complete else None,independent_choices_counts_and_intervals=complete,architecture_review_due=complete))
    header='# Актуально — '+now().strftime('%d.%m.%Y %H:%M')+' MSK\n\n'
    header+='Региональный декодер z13_region_readout_v1 завершён. '+fields['obtained']+' '+fields['decision']
    header+=' Итог независимо проверен; публикацию подтверждает region_readout_recovery1_final_publication.json для recovery1 (либо region_readout_final_publication.json для исходного запуска). При полном результате требуется объединённое обновление однолистовой архитектуры и визуальная проверка новой ветви; генераторы сохранять. Готового следующего научного исполнителя после этой версии пока нет. При ошибке проверять исходные журналы и бюджет, не повторять готовое. Авторизация до26.10 07:40МСК сохраняется; 57 чужих графправок и .gitignore не трогать.\n\nНиже прежние передачи сохранены.\n\n'
    path=W/'HANDOFF.md';path.write_text(header+path.read_text(encoding='utf-8'),encoding='utf-8')


def np_mean(values):
    # Match NumPy reduction order used by the worker, rather than change tie rules.
    import numpy as np
    return np.mean(values)


def publish():
    from scripts import z13_publish_assignment_effect_20261024 as publisher
    publisher.RUN=RUN
    extra=[Path(__file__).relative_to(ROOT).as_posix(),'scripts/z13_region_readout_20261025.py',
        'scripts/z13_region_readout_console.ps1','tracking/z13_region_readout.py','tests/test_z13_region_readout.py']
    if RUN!=ORIGINAL_RUN:
        extra+=['scripts/z13_region_readout_recovery1_20261025.py','scripts/z13_region_readout_recovery1_console.ps1']
        for name in ('protocol.json','controller.json','controller_result.json','launch.json'):
            extra.append((ORIGINAL_RUN/name).relative_to(ROOT).as_posix())
    for name in ('protocol.json','report.json','controller.json','controller_result.json','launch.json',
                 'completion.json','price.json','choice.json','reproduction.json','finish_recorded.json','executed_replay_source.txt','executed_worker_source.txt','crop_binding.json'):
        if (RUN/name).is_file():extra.append((RUN/name).relative_to(ROOT).as_posix())
    extra += [p.relative_to(ROOT).as_posix() for p in (RUN/'videos').glob('*.json')]
    source=inspect.getsource(publisher.publish)
    source=source.replace("names=['experiments/project_history.json'","names=EXTRA+['experiments/project_history.json'")
    source=source.replace("'tests/test_z13_assignment_effect.py','-q'","'tests/test_z13_assignment_effect.py','tests/test_z13_region_readout.py','-q'")
    prefix='region_readout_recovery1_final' if RUN!=ORIGINAL_RUN else 'region_readout_final'
    source=source.replace("prefix='assignment_effect_'+mode",'prefix='+repr(prefix))
    source=source.replace("title='Correct invalid crop-sampling premise and start exact assignment-effect supervision' if mode=='start' else 'Record complete assignment-effect supervision with exact baseline replay'","title='Test detection-region FlyVis readout on unchanged complete assignments'")
    namespace={**vars(publisher),'EXTRA':extra};exec(compile(source,str(Path(__file__)),'exec'),namespace)
    namespace['publish']('region_readout_final')


def main():
    from scripts.z13_hota_deadline_guard import WindowsProcesses
    p=read(RUN/'protocol.json');processes=WindowsProcesses()
    write(RUN/'completion.json',dict(pid=os.getpid(),birth=processes.creation(os.getpid()),started=now().isoformat()))
    while not (RUN/'controller_result.json').exists():
        if now()>datetime.fromisoformat(p['hard_stop'])+timedelta(seconds=90):
            raise TimeoutError('Controller result missing after fixed deadline')
        if (RUN/'controller.json').exists():
            controller=read(RUN/'controller.json')
            if processes.creation(controller['pid'])!=controller['birth']:
                time.sleep(2)
                assert (RUN/'controller_result.json').exists(),'Controller ended without its result'
        time.sleep(3)
    record()
    publish()


if __name__=='__main__':
    try:main()
    except Exception as error:
        write(RUN/'monitor_failure.json',dict(at=now().isoformat(),error=repr(error)))
        raise
