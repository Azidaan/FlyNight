"""Record and publish only this fixed CPU experiment; no recurring chat automation."""
from pathlib import Path
from datetime import datetime,timedelta
import argparse
import subprocess
import time
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_center_hota_20261021 import read,write,sha,now
from scripts.z13_record_center_hota_20261021 import revise
from scripts.z13_hota_deadline_guard import WindowsProcesses
from scripts.z13_continue import paired,per_video_useful
RUN=ROOT/'runs/z13_hinge_target_20261023';IDENT='z13_hinge_target_v1'


def git(*args):return subprocess.check_output(['git','--no-optional-locks',*args],cwd=ROOT).decode('utf-8').strip()


def journal(title,body):
    from scripts.z13_hinge_target_20261023 import journal as record
    record(title,body)


def graph_owned():
    report=RUN/'vault_finish.json' if (RUN/'vault_finish.json').exists() else RUN/'vault_prepare.json'
    if not report.exists():report=ROOT/'runs/z13_boundary_readout_20261022/vault_finish.json'
    foreign={name.replace('\\','/') for name in read(report)['protected_existing_edits']}
    # Our prior completed card was officially moved out of 09; its old receipt
    # proves ownership. Stage that deletion rather than preserving a stale plan.
    prior=read(ROOT/'runs/z13_balanced_ridge_20261022/vault_finish.json')['refreshed_owned_notes']
    old_plan='ObsidianFlyVis/09 Запланировано/z13_balanced_ridge_v1.md'
    if old_plan in {name.replace('\\','/') for name in prior} and not (ROOT/old_plan).exists():foreign.discard(old_plan)
    names=set()
    for args in (['diff','--name-only','-z','HEAD','--','ObsidianFlyVis'],['ls-files','--others','--exclude-standard','-z','--','ObsidianFlyVis']):names.update(name for name in git(*args).split('\0') if name)
    return names-foreign


def manifest():
    files=['experiments/project_history.json','dissertation/ЕДИНЫЙ_ПЛАН.md','tracking/z13_hinge_target.py','tests/test_z13_hinge_target.py','scripts/z13_hinge_target_20261023.py','scripts/z13_hinge_records_20261023.py','scripts/z13_hinge_progress_20261023.ps1','scripts/build_flyvis_architecture.py','scripts/flyvis_architecture_layout.py','scripts/obsidian_fly_problems.py']
    files.extend(graph_owned())
    files.extend(path.relative_to(ROOT).as_posix() for path in RUN.glob('*.json') if path.name not in ('status.json','publication_manifest.json','publish_result.json','watch_launch.json'))
    for arm in ('D','C','R1'):files.extend(path.relative_to(ROOT).as_posix() for path in (RUN/arm).glob('*.json'))
    files.extend(path.relative_to(ROOT).as_posix() for path in (ROOT/'dissertation/architecture').glob('*') if path.name in ('FlyVis_architecture.drawio','FlyVis_architecture.svg','FlyVis_architecture.png','FlyVis_architecture_preview.png','FlyVis_architecture_detail.png','FlyVis_architecture_sources.json','validation.json'))
    missing={name for name in graph_owned() if not (ROOT/name).exists()}
    write(RUN/'publication_manifest.json',{name:sha(ROOT/name) if (ROOT/name).is_file() else None for name in sorted(set(files)|missing) if (ROOT/name).is_file() or name in missing})


def running():
    technical=read(RUN/'technical.json');assert technical['proceed'];cat=read(ROOT/'experiments/project_history.json');card=next(c for c in cat['experiments'] if c['id']==IDENT);assert card['state']=='ready';card['state']='running';write(ROOT/'experiments/project_history.json',cat)
    journal('проба решателя и контроля пройдена, начат полный train26','*Получено.* '+str(technical)+'\n\nИсточник: runs/z13_hinge_target_20261023/technical.json. Это стоимость и сходимость, не оценка эффективности8эпизодов. Первый фрагмент сохранён и не повторяется; выполняются все3варианта,26train. Контроллер: runs/z13_hinge_target_20261023/controller.json.\n\nПредыдущая публикация взвешенной модели d823b9c отправлена после исчезновения временной блокировки Git; index.lock не удалялся. Старый record_blocked.json сохранён как история сбоя, актуальный успех — runs/z13_balanced_ridge_20261022/publish_result.json.')


def finish():
    assert not (RUN/'finish_recorded.json').exists();end=read(RUN/'controller_result.json');controller=read(RUN/'controller.json');ps=WindowsProcesses()
    assert end['worker_ended'] and end['all_owned_ended'] and not end['remaining'] and ps.creation(controller['pid'])!=controller['birth']
    assert not any(ps.creation(int(pid))==birth for pid,birth in end['owned'].items())
    assert not git('diff','--name-only','HEAD','--','experiments/project_history.json','dissertation/ЕДИНЫЙ_ПЛАН.md'),'Shared records changed; preserve them'
    p=read(RUN/'protocol.json')
    for name,digest in p['sha256'].items():assert sha(ROOT/name)==digest,name
    complete=end['exit_code']==0 and not end['hard_timeout'] and (RUN/'report.json').exists() and all((RUN/arm/'done.json').exists() for arm in p['arms'])
    cat=read(ROOT/'experiments/project_history.json');card=next(c for c in cat['experiments'] if c['id']==IDENT);head=git('rev-parse','HEAD');confirmed=[]
    if complete:
        from scripts.z13_score_swap import key,selected_keys
        r=read(RUN/'report.json');pv={};spv={};rep=read(RUN/'reproduction.json')
        for arm in p['arms']:
            done=read(RUN/arm/'done.json');assert rep[arm]['rows']==8326 and rep[arm]['max_center_px']<.5 and rep[arm]['max_score']<.001 and rep[arm]['solver_max_gradient_ratio']<=1 and rep[arm]['solver_max_iterations']<=100
            assert rep[arm]['lambda_value']==read(ROOT/f'runs/z13_target_center_20261018/{arm}/choice.json')['center']['lambda_value']
            for mode in ('balanced','hinge'):
                path=RUN/arm/(mode+'_rows.json');assert sha(path)==done['sha256'][mode];rows=read(path);assert len(rows)==8326;name=arm+'_'+mode;pv[name]=per_video_useful(rows);chosen=selected_keys(rows,.01);assert len(chosen)==84;spv[name]=per_video_useful([dict(row,useful=bool(row.get('useful')) and key(row) in chosen) for row in rows])
        assert pv==r['per_video'] and spv==r['selected_per_video']
        for name,obj in r['paired'].items():a,b=name.split('-');assert paired(pv[a],pv[b])==obj,name
        for name,obj in r['selected_paired'].items():a,b=name.split('-');assert paired(spv[a],spv[b])==obj,name
        d=r['paired']['D_hinge-D_balanced'];selection=r['selected_paired']['D_hinge-D_balanced'];fly=r['paired']['D_hinge-R1_hinge']
        criteria=dict(reproduction=True,localization=d['mean_per_video']>=p['minimum_useful_shift'] and d['ci95'][0]>0,selection=selection['ci95'][0]>0,fly_control=fly['ci95'][0]>0)
        assert criteria==r['criteria'] and all(criteria.values())==r['primary_pass'];negative=d['ci95'][1]<p['minimum_useful_shift'] or selection['ci95'][1]<=0 or fly['ci95'][1]<=0
        state='completed_positive' if r['primary_pass'] else ('completed_negative' if negative else 'inconclusive');acceptance=list(criteria.values())
        obtained='runs/z13_hinge_target_20261023/report.json → paired='+str(r['paired'])+'; selected_paired='+str(r['selected_paired'])+'; summary='+str(r['summary'])+'. Полные D/C/R1,8326строк каждый; все источники/сходимость/исходный контроль проверены.'
        interpretation='Все поисковые критерии выполнены; это ещё не перенос и не HOTA.' if r['primary_pass'] else 'Общие критерии не выполнены; эта односторонняя ошибка не принята. Частные сравнения сохраняются отдельно; по этому исходу не закрывается линия FlyVis.'
        card['metrics']=[dict(key='useful_center_fraction',value=r['summary']['D_hinge']['useful_mean_video'],baseline=r['summary']['D_balanced']['useful_mean_video'],subset='train26;D hinge vs balanced',source='runs/z13_hinge_target_20261023/report.json',locator='summary.D_hinge.useful_mean_video')]
        for name,obj in r['paired'].items():
            if name.startswith(('D_hinge-','C_hinge-')) and obj['ci95'][0]>0:confirmed.append(dict(id=IDENT+'_'+name.replace('-','_vs_'),title='Односторонняя ошибка: '+name,category='readout',metric='useful_center_fraction',unit='fraction',comparison=name,subset='Повторно используемые26train-видео,все известные присутствующие цели',conditions='Тот же гауссов центр,баланс0,5/0,5,история,lambda и ядро; меняется только остаточная ошибка фона; парный бутстрэп5000,сид1.',limitations='Поисковое сравнение,не HOTA,не калиброванное отсутствие,не доказательство анатомии. Частный положительный эффект не отменяет остальных критериев.',evidence=dict(source='runs/z13_hinge_target_20261023/report.json',sha256=sha(RUN/'report.json'),locator=['paired',name])))
    else:
        state='infrastructure';acceptance=[None]*4;obtained='runs/z13_hinge_target_20261023/controller_result.json → '+str(end)+'; завершены '+str([arm for arm in p['arms'] if (RUN/arm/'done.json').exists()])+'. Полного сравнения нет.';interpretation='Сохранён частичный технический исход; эффективность не оценена. Завершённые фрагменты и варианты не повторять.'
    for field,value in [('summary',interpretation),('obtained',obtained),('interpretation',interpretation),('decision',p['decision'] if complete else 'Сохранить остаток, разобрать техническую причину, не выносить вердикт по неполному сравнению.')]:revise(card,field,value,head,'Проверенный полный либо частичный CPU-исход.')
    card.update(state=state,done=f'{now():%Y-%m-%d}',confirmed_results=confirmed);card['commits']=list(dict.fromkeys(card.get('commits',[])+[head]))
    for criterion,value in zip(card['acceptance'],acceptance):criterion.update(met=value,evidence='runs/z13_hinge_target_20261023/report.json' if complete else 'runs/z13_hinge_target_20261023/controller_result.json')
    card['evidence'].append(dict(kind='file',ref='runs/z13_hinge_target_20261023/controller_result.json',claim='Фактическое окончание своего дерева'))
    if complete:card['evidence'].append(dict(kind='file',ref='runs/z13_hinge_target_20261023/report.json',claim='Полный первичный исход и парные контроли'))
    write(ROOT/'experiments/project_history.json',cat);journal('сохранён исход односторонней ошибки фона','*Получено.* '+obtained+'\n\n*Вывод.* '+interpretation+'\n\n*Решение.* '+card['decision']+'\n\nFlyVis — объект; изменён узел подгонки считывания. Все формулировки относятся к выполненному протоколу; прежние версии сохранены. HOTA не измерялась, процент к стенду и доля потолка не определены.');write(RUN/'finish_recorded.json',dict(at=now().isoformat(),complete=complete,state=state,commit_before=head,all_owned_ended=True,report_sha256=sha(RUN/'report.json') if complete else None))


def checks_graph():
    codes=[]
    for i,args in enumerate((['scripts/validate_project_history.py'],['-m','pytest','tests/test_project_history.py','tests/test_z13_hinge_target.py','-q'])):
        with (RUN/f'check_{i}.log').open('w',encoding='utf-8') as log:code=subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
        codes.append(code);assert code==0,args
    if read(RUN/'finish_recorded.json')['complete']:
        assert not git('diff','--name-only','HEAD','--','dissertation/architecture','scripts/build_flyvis_architecture.py','scripts/flyvis_architecture_layout.py'),'Architecture edits must be preserved'
        subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'scripts/build_flyvis_architecture.py'],cwd=ROOT,check=True);journal('архитектурная веха: проверена односторонняя ошибка','На единственном листе добавлена отдельная замена функции ошибки фона с первичным источником. Критерии и ограничения сохранены; источник структурной проверки — dissertation/architecture/validation.json. Новых расчётов ради оформления не было.')
    previous=read(RUN/'publication_manifest.json');receipt={name:digest for name,digest in previous.items() if name.startswith('ObsidianFlyVis/') and name.endswith('.md') and (ROOT/name).is_file() and sha(ROOT/name)==digest};write(RUN/'owned_graph_receipt.json',receipt)
    subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'scripts/z13_build_vault_preserving_edits.py','--report',str(RUN/'vault_finish.json'),'--refresh-owned',str(RUN/'owned_graph_receipt.json')],cwd=ROOT,check=True)
    write(RUN/'checks.json',dict(at=now().isoformat(),codes=codes,graph_completed=True,catalog_sha256=sha(ROOT/'experiments/project_history.json'),plan_sha256=sha(ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md')))


def publish(message,final=True):
    assert not git('diff','--cached','--name-only'),'Existing staged changes preserved'
    for attempt in range(13):
        if not (ROOT/'.git/index.lock').exists():break
        time.sleep(5)
    else:raise RuntimeError('Git lock still present after60s; no deletion or repeated publication')
    previous=read(RUN/'publication_manifest.json')
    for name,digest in previous.items():
        if name.startswith(('scripts/','tracking/','tests/')):assert sha(ROOT/name)==digest,'Source changed since reviewed publication: '+name
    if (RUN/'finish_recorded.json').exists():
        checks=read(RUN/'checks.json');assert checks['catalog_sha256']==sha(ROOT/'experiments/project_history.json') and checks['plan_sha256']==sha(ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md'),'Shared records changed after validation'
    manifest();receipt=RUN/'publication_manifest.json';files=read(receipt)
    assert '.gitignore' not in files and not any(Path(name).suffix in ('.pt','.npz','.mov','.mp4','.joblib') for name in files)
    assert all((sha(ROOT/name)==digest if digest is not None else not (ROOT/name).exists()) for name,digest in files.items());allowed=list(files)+[receipt.relative_to(ROOT).as_posix()]
    assert git('branch','--show-current')=='main' and git('remote','get-url','origin')=='https://gitlab.com/aziteam-group/drbdiss.git'
    subprocess.run(['git','-c','gc.auto=0','add','-f','--',*allowed],cwd=ROOT,check=True);staged={name for name in git('diff','--cached','--name-only','-z').split('\0') if name};assert staged<=set(allowed)
    if staged:
        body=RUN/'commit_message.txt';body.write_text(message+'\n\nCo-authored-by: Codex <noreply@openai.com>\n',encoding='utf-8');subprocess.run(['git','-c','gc.auto=0','commit','--file',str(body)],cwd=ROOT,check=True)
    subprocess.run(['git','-c','gc.auto=0','push','origin','main'],cwd=ROOT,check=True);write(RUN/('publish_result.json' if final else 'preparation_publish_result.json'),dict(at=now().isoformat(),phase='finish' if final else 'preparation',commit=git('rev-parse','HEAD'),origin_main=git('rev-parse','origin/main'),files=sorted(staged)))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--running',action='store_true');ap.add_argument('--manifest',action='store_true');ap.add_argument('--watch',action='store_true');ap.add_argument('--publish',action='store_true');ap.add_argument('--prepare-publish',action='store_true');args=ap.parse_args()
    if args.running:running();return
    if args.manifest:manifest();return
    if args.prepare_publish:publish('Prepare fixed one-sided target-model residual and matched balanced controls',final=False);return
    if args.watch:
        deadline=datetime.fromisoformat(read(RUN/'controller.json')['hard_stop'])+timedelta(seconds=40)
        while not (RUN/'controller_result.json').exists():
            if now()>deadline:write(RUN/'record_blocked.json',dict(at=now().isoformat(),reason='Controller did not save end result; no new computation started'));return
            time.sleep(5)
    try:
        finish();checks_graph()
        if args.publish:publish('Record complete fixed one-sided target-model experiment')
    except Exception as error:write(RUN/'record_blocked.json',dict(at=now().isoformat(),reason=repr(error)));raise


if __name__=='__main__':main()
