"""Finite local recording/verification and exact-file publication of authorized runs."""
import argparse
from datetime import datetime,timedelta,timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.z13_parallel_next import read,write,sha
BASE=ROOT/'runs/z13_parallel_next_20261018'
ALIGNED=ROOT/'runs/z13_aligned_core_20261018'
OWN_FILES=['tracking/z13_target_center_torch.py','tracking/z13_utility.py','tracking/z13_aligned_loss.py',
    'scripts/z13_parallel_next.py','scripts/z13_next_controller.py','scripts/z13_next_progress.ps1',
    'scripts/z13_prepare_next.py','scripts/z13_aligned_core.py','scripts/z13_aligned_controller.py',
    'scripts/z13_prepare_aligned.py','scripts/z13_record_next.py','scripts/open_z13_next_progress.cmd',
    'tests/test_z13_next_gates.py','tests/test_z13_aligned_loss.py']


def journal(topic,text):
    p=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md'; content=p.read_text(encoding='utf-8'); anchor='## Журнал изменений\n'
    now=datetime.now(timezone(timedelta(hours=3))); assert anchor in content
    p.write_text(content.replace(anchor,anchor+f'\n**{now:%d.%m.%Y, %H:%M} MSK** — {topic}\n\n'+text+'\n',1),encoding='utf-8')


def update(card,fields,reason):
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
    for field in ('summary','obtained','interpretation','decision','cannot_claim'):
        if field in fields and card.get(field) and card[field]!=fields[field]:
            card.setdefault('revisions',[]).append(dict(date=datetime.now().date().isoformat(),commit=commit,
                  field=field,old=card[field],reason=reason))
    card.update(fields)


def record_cpu(catalog):
    if (BASE/'recorded.json').exists():
        return
    for rel,digest in read(BASE/'protocol.json')['sha256'].items():
        assert sha(ROOT/rel)==digest,rel
    for kind in ('utility','differentiable'):
        folder=BASE/kind; end=read(folder/'controller_result.json')
        assert end['worker_ended'] and end['exit_code']==0 and not end['timed_out']
        report=read(folder/'report.json'); card=next(c for c in catalog['experiments'] if c['id']=='z13_next_'+kind+'_v1')
        source=f'runs/z13_parallel_next_20261018/{kind}/report.json'
        if kind=='utility':
            summary=report['arms']['D']['coverage']
            obtained=source+' (arms.D.coverage): '+json.dumps(summary,ensure_ascii=False)
            interpretation='Ограниченный оценщик по геометрии и одному баллу не прошёл основной критерий; преимущество над геометрией не доказано. Полные карты ответа не проверялись.'
            state='completed_positive' if report['primary_pass'] else 'inconclusive'
            met=[True,report['primary_pass']]
        else:
            obtained=source+': '+json.dumps(report,ensure_ascii=False)
            interpretation='Техническое совпадение и градиент проверены на реальном фрагменте D; ни одного обновления весов. Это готовность реализации, не улучшение научных метрик.'
            state='infrastructure'; met=[report['primary_pass']]
        update(card,dict(state=state,done=datetime.now().date().isoformat(),obtained=obtained,interpretation=interpretation,
              decision=report['decision'],cannot_claim='Нет HOTA, доказательства пользы стенду, гарантии риска на неизвестных исходах или вывода о неработоспособности FlyVis.'),'Завершена разрешённая проверка')
        for item,value in zip(card['acceptance'],met):
            item.update(met=value,evidence=source)
        card['evidence'].append(dict(kind='json',ref=source,claim='Полный первичный результат'))
        journal('итог '+card['id'], '*Получено.* '+obtained+'\n\n*Вывод.* '+interpretation+'\n\n*Решение.* '+report['decision']+'. HOTA не измерялась.')
    journal('служебный запуск параллельных проверок FlyVis',
      'runs/z13_parallel_next_20261018/launch.json и */controller_result.json: две CPU-проверки реально запущены параллельно и завершены. Ошибка неподдерживаемого аргумента Priority была исправлена до создания процессов; запись launcher_parameter_error.json сохранена. Процессы отдельных консолей созданы, но видимость окон на рабочем столе не подтверждена; scripts/open_z13_next_progress.cmd открывает их вручную. Автоматизация чата не включалась.')
    write(BASE/'recorded.json',dict(recorded=True))


def record_aligned(catalog):
    if (ALIGNED/'recorded.json').exists():
        return
    ending=read(ALIGNED/'controller_result.json'); assert ending['worker_ended']
    card=next(c for c in catalog['experiments'] if c['id']=='z13_aligned_core_v1')
    source='runs/z13_aligned_core_20261018/report.json'
    reproduction=read(ALIGNED/'reproduction.json') if (ALIGNED/'reproduction.json').exists() else {}
    if (ALIGNED/'report.json').exists() and ending['exit_code']==0 and not ending['hard_timeout']:
        report=read(ALIGNED/'report.json'); full=report.get('stages',{}).get('1000')
        passed=bool(full and full['primary_pass'])
        state='completed_positive' if passed else 'inconclusive'
        obtained=source+': '+json.dumps(report,ensure_ascii=False)
        interpretation=('Полный критерий центров на train26 выполнен; состояние dev приведено отдельно.' if passed else
                        'Заранее записанный критерий продолжения или полного преимущества не выполнен; эта реализация не подтверждена. Это не закрывает линию FlyVis.')
        decision=report['decision']; acceptance=[bool(reproduction.get('passed')),passed if full else None]
    else:
        state='inconclusive'; obtained='runs/z13_aligned_core_20261018/controller_result.json: '+json.dumps(ending,ensure_ascii=False)
        if (ALIGNED/'checkpoint.pt').exists():
            import torch
            checkpoint=torch.load(ALIGNED/'checkpoint.pt',map_location='cpu',weights_only=False)
            obtained+='; сохранён шаг '+str(checkpoint['step'])+' (checkpoint.pt).'
        interpretation='Сохранён частичный расчёт; вердикт об эффективности не делается.'
        decision='save_partial_without_scientific_verdict'; acceptance=[True if reproduction.get('passed') else None,None]
    update(card,dict(state=state,done=datetime.now().date().isoformat(),obtained=obtained,interpretation=interpretation,decision=decision,
      cannot_claim='Нет HOTA, проверки отложенных75 или причинного доказательства, что именно согласование объясняло прежний разрыв D/C. Один сид; train26 и dev повторно использованы.'),'Завершение разрешённого дневного окна/прогона')
    for item,met in zip(card['acceptance'],acceptance):
        item.update(met=met,evidence=source if (ALIGNED/'report.json').exists() else 'runs/z13_aligned_core_20261018/controller_result.json')
    card['evidence'].append(dict(kind='json',ref='runs/z13_aligned_core_20261018/controller_result.json',claim='Реальное окончание своего дерева'))
    if (ALIGNED/'report.json').exists():
        card['evidence'].append(dict(kind='json',ref=source,claim='Результаты фиксированных ступеней'))
    journal('итог/сохранение z13_aligned_core_v1','*Получено.* '+obtained+'\n\n*Вывод.* '+interpretation+'\n\n*Решение.* '+decision+'. HOTA и75 отложенных не запускались.')
    write(ALIGNED/'recorded.json',dict(recorded=True))


def validate_and_publish(folder,publish,aligned):
    commands=[['scripts/validate_project_history.py'],['-m','pytest','tests/test_project_history.py',
                'tests/test_z13_next_gates.py','tests/test_z13_aligned_loss.py','-q']]
    checks=[]
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    for i,args in enumerate(commands):
        with (folder/f'validation_{i}.log').open('w',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,*args],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300)
        checks.append(result.returncode)
        if result.returncode:
            write(folder/'validation_result.json',dict(exit_codes=checks,accepted=False))
            raise RuntimeError('Catalog/tests failed; no graph publication or commit')
    write(folder/'validation_result.json',dict(exit_codes=checks,accepted=True))
    graph=folder/'vault_preservation.json'
    subprocess.run([sys.executable,'scripts/z13_build_vault_preserving_edits.py','--report',str(graph)],cwd=ROOT,check=True)
    protected={x.replace(chr(92),'/') for x in read(graph)['protected_existing_edits']}
    changed=set()
    for command in (['git','diff','--name-only','-z','HEAD','--','ObsidianFlyVis'],
                    ['git','ls-files','--others','--exclude-standard','-z','--','ObsidianFlyVis']):
        changed.update(x for x in subprocess.check_output(command,cwd=ROOT).decode('utf-8').split('\0') if x)
    paths=['experiments/project_history.json','dissertation/ЕДИНЫЙ_ПЛАН.md']+OWN_FILES+sorted(changed-protected)
    directories=[BASE]+([ALIGNED] if aligned else [])
    for directory in directories:
        paths += [p.relative_to(ROOT).as_posix() for p in directory.rglob('*') if p.is_file() and p.suffix not in ('.next','.tmp')]
    paths=sorted(set(p for p in paths if (ROOT/p).exists()))
    write(folder/'commit_scope.json',dict(paths=paths,protected_graph=sorted(protected)))
    paths.append((folder/'commit_scope.json').relative_to(ROOT).as_posix())
    if not publish:
        print('Recorded and verified; publication not requested for this invocation.');return
    staged=subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=ROOT)
    if staged:
        write(folder/'publish_result.json',dict(published=False,reason='Preexisting staged files; not touched'))
        raise RuntimeError('Preexisting staged files prevented exact-file publication')
    subprocess.run(['git','add','-f','--',*paths],cwd=ROOT,check=True)
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
    if subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT).returncode:
        subprocess.run(['git','commit','-m','Record bounded FlyVis utility and aligned-core checks',
                        '-m','Co-authored-by: Codex <codex@openai.com>'],cwd=ROOT,check=True)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
    result=subprocess.run(['git','push','origin','main'],cwd=ROOT,capture_output=True,text=True,timeout=90)
    write(folder/'publish_result.json',dict(commit=head,published=result.returncode==0,exit_code=result.returncode,
          message=(result.stdout+result.stderr)[-3000:]))
    print('Recorded and verified; commit',head,'push exit',result.returncode)
    if result.returncode:
        raise RuntimeError('Commit saved locally but push failed; inspect publish_result.json')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--aligned',action='store_true');ap.add_argument('--publish',action='store_true');args=ap.parse_args()
    catalog_path=ROOT/'experiments/project_history.json';catalog=read(catalog_path)
    record_cpu(catalog)
    if args.aligned:
        record_aligned(catalog)
    catalog_path.write_text(json.dumps(catalog,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    validate_and_publish(ALIGNED if args.aligned else BASE,args.publish,args.aligned)


if __name__=='__main__':
    main()
