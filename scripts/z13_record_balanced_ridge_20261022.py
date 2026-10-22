"""Bounded background completion recording; no chat heartbeat and no new experiment."""
from pathlib import Path
from datetime import datetime,timedelta
import argparse
import subprocess
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_center_hota_20261021 import read,write,sha,now
from scripts.z13_record_center_hota_20261021 import revise
from scripts.z13_continue import paired,per_video_useful
from scripts.z13_hota_deadline_guard import WindowsProcesses
RUN=ROOT/'runs/z13_balanced_ridge_20261022';CONTROL=RUN/'recovery_cpu';IDENT='z13_balanced_ridge_v1'


def git(*args):return subprocess.check_output(['git','--no-optional-locks',*args],cwd=ROOT).decode('utf-8').strip()


def journal(title,body):
    path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';text=path.read_text(encoding='utf-8');anchor='## Журнал изменений\n';assert text.count(anchor)==1
    path.write_text(text.replace(anchor,anchor+f'\n**{now():%d.%m.%Y, %H:%M} MSK** — {title} `{IDENT}`\n\n'+body+'\n',1),encoding='utf-8')


def running():
    p=read(RUN/'protocol.json');tech=read(RUN/'technical.json');assert tech['proceed']
    cat=read(ROOT/'experiments/project_history.json');card=next(c for c in cat['experiments'] if c['id']==IDENT);assert card['state']=='ready'
    card['state']='running';write(ROOT/'experiments/project_history.json',cat)
    journal('техническая проба пройдена, полный train26 начат',f"*Получено.* CPU,8эпизодов D; прежняя модель воспроизведена: максимум ошибки центра {tech['max_center_px']}, балла {tech['max_score']} при неизменной lambda. Источник: runs/z13_balanced_ridge_20261022/technical.json. Оценка оставшейся стоимости {tech['estimated_remaining_seconds']}с — приближённая, не обещанный срок. Проба сохранена первым фрагментом и не повторяется. Новый контроллер сохраняет {read(CONTROL/'controller.json')['save_at']}; это исходный общий бюджет.\n\n*Вывод.* Пройдена техническая проверка, выводов об эффективности по8эпизодам нет.\n\n*Решение.* Выполняется полный D/C/R1 на26train по исходным критериям, без GPU/dev/HOTA/обучения ядра.")


def finish():
    assert not (RUN/'full_finish_recorded.json').exists()
    end=read(CONTROL/'controller_result.json');control=read(CONTROL/'controller.json');ps=WindowsProcesses()
    assert end['worker_ended'] and end['all_owned_ended'] and not end['remaining']
    assert ps.creation(control['pid'])!=control['birth'] and not any(ps.creation(int(pid))==birth for pid,birth in end['owned'].items())
    # Do not overwrite or stage another actor's edits to shared records.
    assert not git('diff','--name-only','HEAD','--','experiments/project_history.json','dissertation/ЕДИНЫЙ_ПЛАН.md'),'Shared records have uncommitted changes; preserve them and leave result for review'
    p=read(RUN/'protocol.json');q=read(CONTROL/'protocol.json')
    assert sha(RUN/'protocol.json')==q['original_protocol_sha256']
    for protocol in (p,q):
        for name,digest in protocol['sha256'].items():assert sha(ROOT/name)==digest,name
    cat=read(ROOT/'experiments/project_history.json');card=next(c for c in cat['experiments'] if c['id']==IDENT);head=git('rev-parse','HEAD')
    complete=end['exit_code']==0 and not end['hard_timeout'] and (RUN/'report.json').exists() and all((RUN/arm/'done.json').exists() for arm in p['arms'])
    if complete:
        r=read(RUN/'report.json');rep=read(RUN/'reproduction.json');pv={};spv={}
        from scripts.z13_score_swap import key,selected_keys
        for arm in p['arms']:
            done=read(RUN/arm/'done.json')
            assert sha(RUN/arm/'uniform_rows.json')==done['uniform_rows_sha256'] and sha(RUN/arm/'balanced_rows.json')==done['balanced_rows_sha256']
            assert rep[arm]['max_center_px']<.5 and rep[arm]['max_score']<.001 and rep[arm]['rows']==8326
            assert rep[arm]['lambda_value']==read(ROOT/f'runs/z13_target_center_20261018/{arm}/choice.json')['center']['lambda_value']
            for mode in ('uniform','balanced'):
                rows=read(RUN/arm/(mode+'_rows.json'));assert len(rows)==8326
                name=arm+'_'+mode;pv[name]=per_video_useful(rows);chosen=selected_keys(rows,.01);assert len(chosen)==84
                spv[name]=per_video_useful([dict(row,useful=bool(row.get('useful')) and key(row) in chosen) for row in rows])
        assert pv==r['per_video'] and spv==r['selected_per_video']
        for name,obj in r['paired'].items():a,b=name.split('-');assert paired(pv[a],pv[b])==obj,name
        for name,obj in r['selected_paired'].items():a,b=name.split('-');assert paired(spv[a],spv[b])==obj,name
        d=r['paired']['D_balanced-D_uniform'];sel=r['selected_paired']['D_balanced-D_uniform'];fly=r['paired']['D_balanced-R1_balanced']
        criteria=dict(reproduction=True,localization=d['mean_per_video']>=p['minimum_useful_shift'] and d['ci95'][0]>0,selection=sel['ci95'][0]>0,fly_control=fly['ci95'][0]>0)
        assert criteria==r['criteria'] and all(criteria.values())==r['primary_pass']
        negative=d['ci95'][1]<p['minimum_useful_shift'] or sel['ci95'][1]<=0 or fly['ci95'][1]<=0
        state='completed_positive' if r['primary_pass'] else ('completed_negative' if negative else 'inconclusive')
        obtained='runs/z13_balanced_ridge_20261022/report.json → paired='+str(r['paired'])+'; selected_paired='+str(r['selected_paired'])+'; summary='+str(r['summary'])+'. Все3варианта,8326строк каждый; CPU, неизменные исходники, полный исходный контроль.'
        interpretation='Все критерии поисковой проверки выполнены; требуется отдельная неизменная проверка переноса, это не HOTA.' if r['primary_pass'] else 'Основные критерии не выполнены; фиксированная балансировка не принята. Не означает бесполезность иных функций ошибки или обучения FlyVis.'
        acceptance=[criteria[k] for k in ('reproduction','localization','selection','fly_control')]
        card['metrics']=[dict(key='useful_center_fraction',value=float(np.mean(list(pv['D_balanced'].values()))),baseline=float(np.mean(list(pv['D_uniform'].values()))),subset='train26;D balanced',source='runs/z13_balanced_ridge_20261022/report.json',locator='summary.D_balanced.useful_mean_video')]
        confirmed=[]
        for name,obj in r['paired'].items():
            if name.startswith(('C_balanced-','D_balanced-')) and obj['ci95'][0]>0:
                confirmed.append(dict(id=IDENT+'_'+name.replace('-','_vs_'),title='Взвешенная модель: '+name,category='readout',metric='useful_center_fraction',unit='fraction',comparison=name,subset='Повторно используемые26train-видео, все известные присутствующие цели',conditions='Фиксированные веса цели/фона0,5/0,5; те же признаки, история, lambda и ядро; парный бутстрэп5000,сид1.',limitations='Поисковый train26; частное сравнение не отменяет остальных критериев. Не HOTA, не калиброванный отказ, не доказательство анатомического преимущества.',evidence=dict(source='runs/z13_balanced_ridge_20261022/report.json',sha256=sha(RUN/'report.json'),locator=['paired',name])))
        card['confirmed_results']=confirmed
    else:
        state='infrastructure';acceptance=[None]*4
        obtained='runs/z13_balanced_ridge_20261022/recovery_cpu/controller_result.json → '+str(end)+'; technical.json → '+str(read(RUN/'technical.json') if (RUN/'technical.json').exists() else None)+'. Сохранённые завершённые части: '+str([arm for arm in p['arms'] if (RUN/arm/'done.json').exists()])+'. Полного сравнения нет.'
        interpretation='Сохранён частичный технический исход; эффективность не оценена. Завершённые части не повторять.'
    for field,value in [('summary',interpretation),('obtained',obtained),('interpretation',interpretation),('decision',p['decision'] if complete else 'Сохранить остаток и разобрать служебную причину без вывода об эффективности.')]:revise(card,field,value,head,'Проверенный полный либо частичный исход отдельного CPU-прогона.')
    card.update(state=state,done=f'{now():%Y-%m-%d}')
    for criterion,value in zip(card['acceptance'],acceptance):criterion.update(met=value,evidence='runs/z13_balanced_ridge_20261022/report.json' if complete else 'runs/z13_balanced_ridge_20261022/recovery_cpu/controller_result.json')
    card['commits']=list(dict.fromkeys(card.get('commits',[])+[head]))
    card['evidence'].extend([dict(kind='file',ref='runs/z13_balanced_ridge_20261022/recovery_cpu/protocol.json',claim='Служебное CPU-восстановление в исходном бюджете'),dict(kind='file',ref='runs/z13_balanced_ridge_20261022/recovery_cpu/controller_result.json',claim='Фактическое окончание своего дерева')])
    if complete:card['evidence'].append(dict(kind='file',ref='runs/z13_balanced_ridge_20261022/report.json',claim='Полный первичный исход всех3вариантов'))
    write(ROOT/'experiments/project_history.json',cat)
    journal('сохранён исход взвешенной модели цели','*Получено.* '+obtained+'\n\n*Вывод.* '+interpretation+'\n\n*Решение.* '+card['decision']+'\n\nHOTA не измерялась; процент к стенду и доля потолка не определены. Понятия проверены: FlyVis — объект; взвешенная регрессия — изменённый узел считывания; вывод относится только к выполненному протоколу. Прежние итоговые поля сохранены в revisions.')
    write(RUN/'full_finish_recorded.json',dict(at=now().isoformat(),complete=complete,state=state,commit_before=head,all_owned_ended=True,report_sha256=sha(RUN/'report.json') if complete else None))


def checks_and_graph():
    commands=[['scripts/validate_project_history.py'],['-m','pytest','tests/test_project_history.py','tests/test_z13_balanced_ridge.py','-q']]
    results=[]
    for i,args in enumerate(commands):
        with (RUN/f'finish_check_{i}.log').open('w',encoding='utf-8') as log:code=subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
        results.append(code);assert code==0,args
    if read(RUN/'full_finish_recorded.json')['complete']:
        assert not git('diff','--name-only','HEAD','--','dissertation/architecture','scripts/build_flyvis_architecture.py','scripts/flyvis_architecture_layout.py'),'Author architecture edits present; preserve them'
        subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'scripts/build_flyvis_architecture.py'],cwd=ROOT,check=True)
        journal('архитектурная веха: проверена взвешенная модель','Добавлена самостоятельная ветвь подгонки цели/фона по полному CPU-исходу. Перекрытия, читаемость и единственный лист проверены генератором: dissertation/architecture/validation.json. Это функция ошибки считывания; прежние ветви сохранены, HOTA не пересчитывалась.')
    # Refresh only notes owned in our pre-run publication receipt and still byte-identical.
    manifest=read(ROOT/'runs/z13_absolute_confidence_20261022/publication_manifest.json')
    receipt={name:digest for name,digest in manifest.items() if name.startswith('ObsidianFlyVis/') and name.endswith('.md') and (ROOT/name).exists() and sha(ROOT/name)==digest}
    write(RUN/'owned_graph_receipt.json',receipt)
    subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'scripts/z13_build_vault_preserving_edits.py','--report',str(RUN/'vault_finish.json'),'--refresh-owned',str(RUN/'owned_graph_receipt.json')],cwd=ROOT,check=True)
    write(RUN/'checks.json',dict(at=now().isoformat(),codes=results,graph_completed=True,catalog_sha256=sha(ROOT/'experiments/project_history.json'),plan_sha256=sha(ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md')))


def publish():
    assert not git('diff','--cached','--name-only'),'Existing staged changes must be preserved'
    assert not (ROOT/'.git/index.lock').exists(),'Git lock present; never remove it automatically'
    old=read(ROOT/'runs/z13_absolute_confidence_20261022/publication_manifest.json')
    for name,digest in old.items():
        if name.startswith(('scripts/','tracking/','tests/')):assert sha(ROOT/name)==digest,'Source changed after reviewed publication: '+name
    checks=read(RUN/'checks.json');assert checks['catalog_sha256']==sha(ROOT/'experiments/project_history.json') and checks['plan_sha256']==sha(ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md')
    subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'scripts/z13_evening_records_20261022.py','--manifest'],cwd=ROOT,check=True)
    receipt=ROOT/'runs/z13_absolute_confidence_20261022/publication_manifest.json';manifest=read(receipt)
    for name,digest in manifest.items():assert sha(ROOT/name)==digest,name
    files=list(manifest)+[receipt.relative_to(ROOT).as_posix()]
    assert '.gitignore' not in files and not any(Path(name).suffix in ('.pt','.npz','.mov','.mp4','.joblib') for name in files)
    subprocess.run(['git','-c','gc.auto=0','add','-f','--',*files],cwd=ROOT,check=True)
    staged={name for name in git('diff','--cached','--name-only','-z').split('\0') if name};assert staged<=set(files)
    if staged:
        body='Record complete bounded target/background ridge test\n\nFixed observed-only weighting, original reproduction and paired controls.\n\nCo-authored-by: Codex <noreply@openai.com>\n'
        message=RUN/'commit_message.txt';message.write_text(body,encoding='utf-8')
        subprocess.run(['git','-c','gc.auto=0','commit','--file',str(message)],cwd=ROOT,check=True)
    assert git('branch','--show-current')=='main'
    assert git('remote','get-url','origin') in ('https://gitlab.com/aziteam-group/drbdiss.git','https://gitlab.com/aziteam-group/drbdiss')
    subprocess.run(['git','-c','gc.auto=0','push','origin','main'],cwd=ROOT,check=True)
    write(RUN/'publish_result.json',dict(at=now().isoformat(),commit=git('rev-parse','HEAD'),origin_main=git('rev-parse','origin/main'),files=sorted(staged)))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--running',action='store_true');ap.add_argument('--watch',action='store_true');ap.add_argument('--publish',action='store_true');args=ap.parse_args()
    if args.running:running();return
    if args.watch:
        end=datetime.fromisoformat(read(CONTROL/'protocol.json')['hard_stop'])+timedelta(seconds=40)
        while not (CONTROL/'controller_result.json').exists():
            if now()>end:write(RUN/'record_blocked.json',dict(at=now().isoformat(),reason='Controller result absent after original deadline; no extra computation started'));return
            time.sleep(5)
    try:
        finish();checks_and_graph()
        if args.publish:publish()
    except Exception as error:write(RUN/'record_blocked.json',dict(at=now().isoformat(),reason=repr(error)));raise


if __name__=='__main__':main()
