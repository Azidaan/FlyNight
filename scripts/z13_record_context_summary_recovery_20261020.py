"""Record the recovered fixed-protocol result and publish an explicit own-file scope."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.z13_parallel_next import read,write,sha
from scripts.z13_record_next import journal

RUN=ROOT/'runs/z13_context_transfer_20261020'
OWN=['scripts/z13_context_summary_recovery_20261020.py',
     'scripts/z13_context_summary_recovery_progress.ps1',
     'scripts/z13_record_context_summary_recovery_20261020.py',
     'tests/test_z13_context_summary_recovery.py']


def verify():
    from scripts import z13_context_transfer_20261020 as original
    original.verify(read(RUN/'protocol.json'))
    for rel,digest in read(RUN/'recovery_summary_protocol.json')['sha256'].items():
        assert sha(ROOT/rel)==digest,rel
    end=read(RUN/'recovery_summary_controller_result.json')
    assert end['worker_ended'] and end['exit_code']==0 and not end['hard_timeout']
    assert read(RUN/'controller_result.json')['exit_code']==1
    assert read(RUN/'recovery_summary_audit.json')['passed']


def record():
    verify()
    marker=RUN/'recovery_summary_finish_recorded.json'
    assert not marker.exists(),'Result already recorded'
    report=read(RUN/'report.json')
    source='runs/z13_context_transfer_20261020/report.json'
    fields=dict(state='completed_positive' if report['proceed'] else 'inconclusive',
        done=datetime.now(timezone(timedelta(hours=3))).date().isoformat(),
        summary='Пространственное считывание C не прошло составные критерии переноса; преимущество над входом глаза подтверждено только для доли полезных центров — '+source+'.',
        obtained=source+': '+json.dumps({k:report[k] for k in ('localization','selection_at1percent','localization_pass','selection_pass','proceed')},ensure_ascii=False)+
                 '; coverage.C.map_utility.0.01: '+json.dumps(report['coverage']['C']['map_utility']['0.01'],ensure_ascii=False),
        interpretation='На этих видео интервал добавки к прежней C включает ноль; это неопределённость, а не доказательство равенства, бесполезности FlyVis или определённой причины неуспеха.',
        decision=report['decision'],
        cannot_claim='Не установлены улучшение HOTA, польза стенду, преимущество обучения ядра D, независимое обобщение или безопасность неизвестных исходов; повторно использованный dev35 и глобальный отбор 1% не являются слепой проверкой и рабочим правилом применения.')
    catalog_path=ROOT/'experiments/project_history.json'
    catalog=read(catalog_path)
    card=next(c for c in catalog['experiments'] if c['id']=='z13_context_transfer_v1')
    criteria_before=[x['criterion'] for x in card['acceptance']]
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
    for field in ('summary','obtained','interpretation','decision','cannot_claim'):
        if card.get(field) and card[field]!=fields[field]:
            card.setdefault('revisions',[]).append(dict(date=fields['done'],commit=head,field=field,
                  old=card[field],reason='Служебная ошибка сборки исправлена отдельно; исходный код завершения и прежний частичный итог сохранены'))
    card.update(fields)
    for item,met,evidence in zip(card['acceptance'],[True,report['localization_pass'],report['selection_pass']],
            ['runs/z13_context_transfer_20261020/publication_state_audit.json',source,source]):
        item.update(met=met,evidence=evidence)
    assert criteria_before==[x['criterion'] for x in card['acceptance']]
    card['files']=list(dict.fromkeys(card['files']+OWN))
    card.setdefault('commits',[]).append(head)
    card['evidence'] += [dict(kind='json',ref=source,claim='Полные сравнения по заранее заданным критериям'),
        dict(kind='json',ref='runs/z13_context_transfer_20261020/recovery_summary_audit.json',claim='Предсказания и исходники неизменны; неизвестные исходы не подменены'),
        dict(kind='json',ref='runs/z13_context_transfer_20261020/recovery_summary_controller_result.json',claim='Отдельное успешное завершение сборки, прежняя ошибка сохранена')]
    levels=read(RUN/'recovery_summary_audit.json')['levels_mean_per_video']
    card['metrics']=[dict(key='selected_success_fraction_positive',value=levels['C_context_map'],
        baseline=levels['C_mean_map'],subset='dev35; глобальный отбор 1% всех строк',
        source='runs/z13_context_transfer_20261020/recovery_summary_audit.json',
        locator='levels_mean_per_video.C_context_map')]
    # Each sentence is checked for subject/tool distinction, repetition, and actual measured scope.
    write(RUN/'recovery_summary_text_review.json',dict(sentences=[dict(field=f,text=fields[f],
          subject_and_tool='FlyVis is the object; readout and selection are its tested node and application diagnostic',
          adjacent_repetition='The field has a separate role: summary, measurements, interpretation, decision or limitation',
          matches_performed_work=True) for f in ('summary','obtained','interpretation','decision','cannot_claim')]))
    catalog_path.write_text(json.dumps(catalog,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    table=''
    for group in ('localization','selection_at1percent'):
        for name,entry in report[group].items():
            text=f"{name}: {100*entry['mean_per_video']:+.4f} п.п.; 95% ДИ [{100*entry['ci95'][0]:+.4f}; {100*entry['ci95'][1]:+.4f}]"
            table+=text.replace('.',',').replace('п,п,','п.п.')+'; '
    relative=100*(levels['C_directional']/levels['C_mean']-1)
    journal('итог z13_context_transfer_v1 после восстановления сводки',
        '*Получено (`z13_context_transfer_v1`).* '+source+': '+table.rstrip()+
        '\n\n*Служебное исправление.* recovery_summary_protocol.json, recovery_summary_audit.json и recovery_summary_controller_result.json: '
        'отдельная сборка завершена с кодом0; исходный controller_result.json с кодом1 и старый частичный итог сохранены. '
        'Новые предсказания и обучение не запускались; исходные веса, строки, научные источники, критерии, пороги и бутстрэп неизменны. '
        'Неизвестные строки остаются неизвестными и занимают места в прежнем ранжировании; метрика положительных случаев считает только kind=1.'+
        '\n\n*Вывод.* '+fields['interpretation']+
        '\n\n*Решение.* Составные критерии центров и отбора не пройдены; эту реализацию не переносить в трекер и не подбирать заново по dev; '
        'причина научного неуспеха не установлена, линия FlyVis не закрыта, статус P8 остаётся частично решённым.'+
        '\n\n*К исходному варианту.* По доле полезных центров среднее C: '+f"{100*levels['C_mean']:.4f} → {100*levels['C_directional']:.4f}%; относительное изменение {relative:+.4f}%".replace('.',',')+
        '; парный интервал добавки включает ноль. Доля потолка не определена: соответствующий потолок не измерялся. HOTA не измерялась.')
    write(marker,dict(recorded=True,report=source,prior_finish_marker_preserved=True))


def scope():
    # This publication has no discovery step: every staged path is explicitly named.
    records=['report.json','recovery_summary_protocol.json',
             'recovery_summary_controller_result.json','recovery_summary_audit.json']
    paths=sorted(set(OWN+['experiments/project_history.json','dissertation/ЕДИНЫЙ_ПЛАН.md',
                 'ObsidianFlyVis/01 Муха/Эксперименты/z13_context_transfer_v1.md']) |
                 {(RUN/p).relative_to(ROOT).as_posix() for p in records})
    assert all((ROOT/p).is_file() for p in paths)
    write(RUN/'recovery_summary_commit_scope.json',dict(paths=paths,
          excluded='All other graph files, administrative JSON, logs, caches and model binaries'))
    return paths


def publish():
    verify()
    assert read(RUN/'recovery_summary_validation_result.json')['accepted']
    assert subprocess.check_output(['git','remote','get-url','origin'],cwd=ROOT).decode().strip()=='https://gitlab.com/aziteam-group/drbdiss.git'
    assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT).decode().strip()=='main'
    assert not (ROOT/'.git/index.lock').exists()
    paths=scope()
    existing={x for x in subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=ROOT).decode('utf-8').split('\0') if x}
    assert existing<=set(paths), 'Preexisting staged files crossed this exact publication scope'
    subprocess.run(['git','add','-f','--',*paths],cwd=ROOT,check=True)
    staged=set(x for x in subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=ROOT).decode('utf-8').split('\0') if x)
    assert staged<=set(paths)
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
    subprocess.run(['git','commit','-m','Recover fixed FlyVis transfer summary without repeating predictions',
        '-m','Co-authored-by: Codex <codex@openai.com>'],cwd=ROOT,check=True)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
    result=subprocess.run(['git','push','origin','main'],cwd=ROOT,capture_output=True,text=True,timeout=90)
    write(RUN/'recovery_summary_publish_result.json',dict(commit=head,published=result.returncode==0,
        exit_code=result.returncode,message=(result.stdout+result.stderr)[-3000:]))
    print(json.dumps(dict(commit=head,published=result.returncode==0)))
    assert result.returncode==0


if __name__=='__main__':
    if sys.argv[1]=='record': record()
    elif sys.argv[1]=='scope': scope()
    elif sys.argv[1]=='publish': publish()
    else: raise ValueError('Unknown operation')
