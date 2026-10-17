"""Verify saved paired results and record the full probe, separately from preparation."""
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.z13_continue import per_video_useful, paired, coverage_table
RUN=ROOT/'runs/z13_target_center_20261018'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    marker=RUN/'full_finish_recorded.json'
    if marker.exists():
        raise RuntimeError('Full result already recorded')
    end=read(RUN/'io_recovery_result.json')
    assert end['exit_code']==0 and end['worker_ended'] and not end['hard_timeout'] and end['preserved_D_C_unchanged']
    proto=read(RUN/'protocol.json');report=read(RUN/'report.json')
    for rel,digest in proto['sha256'].items():
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==digest,rel
    saved=read(RUN/'io_recovery_controller.json')['preserved_sha256']
    for rel,digest in saved.items():
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==digest,rel
    rows={}
    for arm in proto['arms']:
        assert read(RUN/arm/'done.json')['source_verified']
        r=read(RUN/arm/'reproduction.json')
        assert r['lambda_matches'] and r['max_center_px']<.5 and r['max_score_difference']<.001
        for mode in ('binary','center'):
            rows[arm+'_'+mode]=read(RUN/arm/(mode+'_rows.json'))
    rows['D_prior']=read(ROOT/'runs/z13_continue_20261017/baseline/rows.json')
    keys=[{(r['label'],r['video'],r['kind']) for r in values} for values in rows.values()]
    assert all(k==keys[0] for k in keys)
    for name,values in rows.items():
        summary=report['summary'][name]
        assert len(values)==len(keys[0])==summary['rows']
        assert sum(r['kind']==1 for r in values)==summary['positives']
        assert sum(bool(r.get('useful')) for r in values if r['kind']==1)==summary['useful']
        assert coverage_table(values)==summary['coverage']
    for name,value in report['paired'].items():
        left,right=name.split('-')
        assert paired(per_video_useful(rows[left]),per_video_useful(rows[right]))==value,name
    assert report['primary_pass']==all(report['paired'][k]['ci95'][0]>0 for k in ('D_center-D_prior','D_center-D_binary'))
    now=datetime.now(timezone(timedelta(hours=3)))
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    source='runs/z13_target_center_20261018/report.json'
    pieces=[]
    for key in report['paired']:
        p=report['paired'][key]
        s=f"{key}: {p['mean_per_video']*100:+.2f} п.п. [{p['ci95'][0]*100:+.2f}; {p['ci95'][1]*100:+.2f}], лучше на {p['better']} из {p['n']} видео"
        pieces.append(s.replace('.',','))
    obtained=source+' (paired): '+'; '.join(pieces)+'. '
    obtained+='Техническое окончание: io_recovery_result.json, exit_code=0, worker_ended=true; все D/C/R1 done.json и reproduction.json проверены. '
    obtained+=source+' (summary.D_center.coverage): верхний 1% — 2 полезных из 64 известных и 20 неизвестных, верхние 5% — 9 из 338 известных и 79 неизвестных. Это ранжирование, не измеренный вред HOTA.'
    interpretation='Заранее заданный поисковый критерий выполнен: новая цель считывателя D лучше и прежнего считывателя D, и binary D по парным повидео ДИ. Улучшение есть также на штатной C; превосходство D над C не подтверждено. Качество локализации улучшилось, надёжный выбор вмешательств пока не получен. Повторно использованный train26, один набор весов.'
    decision='Зафиксировать центрированную цель как кандидата для отдельной проверки. Предложенный следующий шаг — независимая оценка замороженных условий на dev35 с прежним считывателем, C и R1; без подбора параметров на dev. Новый прогон требует отдельного протокола и согласования бюджета. До подтверждения локализации и пригодного правила вмешательства не заменять действующий вариант стенда. Длительное обучение ядра прежним способом не возобновлять.'
    cannot='HOTA не измерена; нет доказательства пользы стенду, независимого подтверждения, превосходства обученной D над штатной C, роли коннектома или полной решённости P8. Поле harmful в старой функции означает известные неполезные случаи, а не доказанный вред трекеру.'
    path=ROOT/'experiments/project_history.json';cat=read(path)
    card=next(c for c in cat['experiments'] if c['id']=='z13_target_center_v1')
    changes=dict(obtained=obtained,interpretation=interpretation,decision=decision,cannot_claim=cannot)
    for field,value in changes.items():
        if card.get(field) and card[field]!=value:
            card.setdefault('revisions',[]).append(dict(date=now.date().isoformat(),commit=head,field=field,old=card[field],reason='Полный результат после проверенного служебного восстановления.'))
        card[field]=value
    card.update(state='completed_positive',done=now.date().isoformat())
    for a in card['acceptance']:
        a.update(met=True,evidence=source+'; D/C/R1/reproduction.json; full_verification.json')
    card['evidence'].append(dict(kind='json',ref=source,claim='Полное парное сравнение и ограничения отбора'))
    path.write_text(json.dumps(cat,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';plan=path.read_text(encoding='utf-8');anchor='## Журнал изменений\n';assert anchor in plan
    entry=f'\n**{now:%d.%m.%Y, %H:%M} MSK** — итог `z13_target_center_v1`\n\n*Получено.* {obtained}\n\n*Вывод.* {interpretation}\n\n*Решение.* {decision}\n\n*Нельзя заявлять.* {cannot}\n'
    path.write_text(plan.replace(anchor,anchor+entry,1),encoding='utf-8')
    verified=dict(checked_at=now.isoformat(),frozen_hashes=len(proto['sha256']),all_arms_complete=True,
        controls_reproduced=True,paired_statistics_reproduced=True,coverage_reproduced=True,
        saved_D_C_unchanged=True,report_sha256=hashlib.sha256((RUN/'report.json').read_bytes()).hexdigest())
    (RUN/'full_verification.json').write_text(json.dumps(verified,indent=2)+'\n',encoding='utf-8')
    marker.write_text(json.dumps(dict(recorded_at=now.isoformat(),primary_pass=report['primary_pass']),indent=2)+'\n',encoding='utf-8')
    print(json.dumps(verified))


if __name__=='__main__':
    main()
