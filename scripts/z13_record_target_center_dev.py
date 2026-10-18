"""Local finite completion recording; never schedules chat/model calls or starts science."""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime,timedelta,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.z13_target_center_dev import RUN,read,write,verify


def record(finish=False):
    marker=RUN/('finish_recorded.json' if finish else 'launch_recorded.json')
    if marker.exists():return
    now=datetime.now(timezone(timedelta(hours=3)))
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    catpath=ROOT/'experiments/project_history.json';catalog=read(catpath)
    card=next(c for c in catalog['experiments'] if c['id']=='z13_target_center_dev_v1')
    if not finish:
        control=read(RUN/'controller.json')
        card['state']='running'
        card['evidence'].append(dict(kind='json',ref='runs/z13_target_center_dev_20261018/controller.json',claim='Фактический CPU-запуск и абсолютный срок остановки'))
        text='*Запущено.* Независимый контроллер runs/z13_target_center_dev_20261018/controller.json; CPU, предел четыре часа и30с завершения. Отдельные результаты по видео, без вызовов модели по расписанию. Локальный конечный наблюдатель запишет итог из сохранённых файлов; он не запускает вычисления заново. Видимое окно PowerShell автоматически открыть не удалось из-за ограничения средства управления Windows; подготовлен scripts/open_z13_target_center_dev_progress.cmd для ручного открытия.'
    else:
        end=read(RUN/'controller_result.json');rep=read(RUN/'report.json')
        assert end['worker_ended'] and end['exit_code']==0 and not end['hard_timeout']
        proto=read(RUN/'protocol.json');verify(proto)
        assert len(proto['videos'])==35
        for v in proto['videos']:
            assert read(RUN/'validation'/(v+'.json'))['passed']
            for arm in ('D','C','R1'): assert read(RUN/arm/v/'done.json')['video']==v
        rows=[]
        for key,p in rep['paired'].items():
            rows.append(f"{key}: {p['mean_per_video']*100:+.2f} п.п. [{p['ci95'][0]*100:+.2f}; {p['ci95'][1]*100:+.2f}], {p['n']} видео".replace('.',','))
        obtained='runs/z13_target_center_dev_20261018/report.json (paired): '+'; '.join(rows)+'. Полнота35видео и метки проверены; controller_result.json exit_code=0, worker_ended=true.'
        passed=rep['primary_pass'] and rep['fly_control_pass']
        changes=dict(obtained=obtained,interpretation='Критерии переносимости '+('выполнены' if passed else 'не выполнены')+'. Это фиксированная проверка на ранее использованном dev35; польза стенду не измерялась.',
            decision=rep['decision']+'. Любая новая правка или проверка стенда требует отдельного протокола; dev-параметры по этому результату не подбирались.',
            cannot_claim='Нет HOTA, проверки отложенных75, слепого подтверждения, доказательства надёжного отказа или пользы стенду. harmful означает известный неполезный исход, не измеренный вред трекингу.')
        for field,value in changes.items():
            if card.get(field) and card[field]!=value:
                card.setdefault('revisions',[]).append(dict(date=now.date().isoformat(),commit=head,field=field,old=card[field],reason='Полный сохранённый итог dev35.'))
            card[field]=value
        card.update(state='completed_positive' if passed else 'inconclusive',done=now.date().isoformat())
        for a,met in zip(card['acceptance'],[True,rep['primary_pass'],rep['fly_control_pass']]):
            a.update(met=met,evidence='runs/z13_target_center_dev_20261018/report.json; validation/; controller_result.json')
        card['evidence'].append(dict(kind='json',ref='runs/z13_target_center_dev_20261018/report.json',claim='Сохранённые итоги фиксированной dev-проверки'))
        text='*Получено (`z13_target_center_dev_v1`).* '+obtained+'\n\n*Вывод.* '+card['interpretation']+'\n\n*Решение.* '+card['decision']
    catpath.write_text(json.dumps(catalog,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';plan=path.read_text(encoding='utf-8');anchor='## Журнал изменений\n';assert anchor in plan
    entry=f'\n**{now:%d.%m.%Y, %H:%M} MSK** — '+('итог' if finish else 'запуск')+' `z13_target_center_dev_v1`\n\n'+text+'\n'
    path.write_text(plan.replace(anchor,anchor+entry,1),encoding='utf-8')
    write(marker,dict(recorded_at=now.isoformat()))


def watch():
    deadline=datetime.fromisoformat(read(RUN/'controller.json')['hard_deadline']).timestamp()+60
    while time.time()<deadline and not (RUN/'controller_result.json').exists():time.sleep(5)
    if not (RUN/'controller_result.json').exists():return
    result=read(RUN/'controller_result.json')
    if result['exit_code']!=0 or not (RUN/'report.json').exists():
        write(RUN/'needs_attention.json',dict(reason='Incomplete execution; preserve artifacts, no efficacy verdict',controller=result))
        return
    record(finish=True)
    commands=[['scripts/validate_project_history.py'],['-m','pytest','tests/test_project_history.py','-q'],
        ['scripts/z13_build_vault_preserving_edits.py','--report',str(RUN/'result_vault_preservation.json')]]
    outcomes=[]
    for i,args in enumerate(commands):
        with (RUN/f'finish_check_{i}.log').open('w',encoding='utf-8') as log:
            r=subprocess.run([sys.executable,*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=300)
        outcomes.append(dict(command=args,exit_code=r.returncode))
        if r.returncode:break
    write(RUN/'postprocess_result.json',dict(checks=outcomes,git_commit_pending=True,chat_calls=0))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--finish',action='store_true');parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    if args.watch:watch()
    else:record(args.finish)
