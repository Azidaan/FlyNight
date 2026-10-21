"""Record launch, technical gate, and complete or partial morning shutdown honestly."""
import argparse
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_center_hota_20261021 import RUN,LEGACY,IDENT,read,write,sha,now,journal


def revise(card,field,value,head,reason):
    old=card.get(field,'')
    if old and old!=value:card.setdefault('revisions',[]).append(dict(date=f'{now():%Y-%m-%d}',commit=head,field=field,old=old,reason=reason))
    card[field]=value


def main():
    a=argparse.ArgumentParser();a.add_argument('--phase',choices=['probe_launch','full_launch','finish'],required=True);x=a.parse_args();marker=RUN/(x.phase+'_recorded.json')
    if marker.exists():raise RuntimeError('Already recorded')
    path=ROOT/'experiments/project_history.json';cat=read(path);card=next(c for c in cat['experiments'] if c['id']==IDENT);head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if x.phase=='probe_launch':
        launch=read(RUN/'probe_launch.json');control=read(RUN/'controller.json');assert control['phase']=='probe'
        card['state']='running';revise(card,'summary','Идёт техническое воспроизведение и замер полной версии; HOTA ещё не измерялась.',head,'Фактический запуск зарегистрированной пробы.')
        for name in ('probe_launch.json','service_contract.json'):card['evidence'].append(dict(kind='file',ref='runs/z13_center_hota_20261021/'+name,claim='Фактический запуск и неизменные источники'))
        journal('запуск технической пробы полного трекера','*Сделано.* `runs/z13_center_hota_20261021/probe_launch.json`: '+str(launch)+'. Независимый контроллер до1200с, `controller.json` задаёт сохранение '+control['save_stop_at']+' и конец дерева '+control['hard_stop_at']+'. Отдельное окно PowerShell24492 подтверждено ненулевым дескриптором. Основные научные настройки не меняются; HOTA/цена ещё не получены. Реальные этапы отражены в status.json и журналах, расписание само по себе не является доказательством работы.')
    elif x.phase=='full_launch':
        technical=read(RUN/'technical.json');probe_result='probe_completion_controller_result.json' if 'execution_completion_contract_sha256' in technical else ('probe_recovery1_controller_result.json' if 'recovery_execution_contract_sha256' in technical else 'probe_controller_result.json');end=read(RUN/probe_result);assert technical['passed'] and end['worker_ended'] and end['exit_code']==0 and not end['hard_timeout']
        if 'execution_completion_contract_sha256' in technical:
            assert technical['execution_completion_contract_sha256']==sha(RUN/'completion_contract.json')
            card['evidence'].append(dict(kind='file',ref='runs/z13_center_hota_20261021/completion_contract.json',claim='Отдельное завершение только неиспользованного вычислительного бюджета пробы'))
        if 'recovery_execution_contract_sha256' in technical:
            assert technical['recovery_execution_contract_sha256']==sha(RUN/'recovery1_contract.json')
            assert read(RUN/'probe_controller_result.json')['exit_code']==1
            for name in ('recovery1_contract.json','recovery1_reproduction.json',probe_result):card['evidence'].append(dict(kind='file',ref='runs/z13_center_hota_20261021/'+name,claim='Отдельное проверенное служебное восстановление; прежняя ошибка сохранена'))
        launch=read(RUN/'full_launch.json');card['state']='running';card['acceptance'][0].update(met=True,evidence='runs/z13_center_hota_20261021/technical.json')
        revise(card,'summary','Технические критерии выполнены; идёт неизменная очередь35видео. Полной HOTA ещё нет.',head,'Полная очередь после технического допуска.')
        for name in ('technical.json','full_launch.json','execution_identity.json'):card['evidence'].append(dict(kind='file',ref='runs/z13_center_hota_20261021/'+name,claim='Технический допуск и полный запуск'))
        journal('технический допуск и запуск очереди полного трекера','*Получено.* `runs/z13_center_hota_20261021/technical.json`: пройдены сохранённые C/D/R1 центры/баллы и решения, совпадение скомпилированного исполнения, полное train-видео без вмешательства и прежняя сшивка всех35видео. Измеренная приблизительная цена полного объёма '+f'{technical["rough_full_seconds"]/3600:.2f}'.replace('.',',')+'ч по одному train-видео; она неопределённа и может превышать оставшееся окно.\n\n*Решение.* Начата очередь по неизменному протоколу: '+str(launch)+'. По каждому видео сохраняются C/D/R1, общая MASA и сшивка, затем отдельные маркеры. После07:35 новые задания не начинаются, только текущее может закончиться до07:39:30; неполная очередь даст только остаток без HOTA-вердикта. Это разрешённый автором ограниченный прогон с возобновлением, а не обещание завершения к утру.')
    else:
        result=read(RUN/'full_controller_result.json');assert result['all_own_workers_ended'] and not result['remaining_owned_processes']
        complete=result['state']=='completed' and not result['remaining']
        if complete:
            report=read(RUN/'eval/report.json');old=read(LEGACY/'eval/report.json');assert report['videos']==35
            for new,previous in [('A_masa','A_masa'),('legacyC','C_masa'),('legacyD','D_masa')]:assert abs(report['pooled'][new+'|all']['HOTA']-old['pooled'][previous+'|all']['HOTA'])<1e-10
            import numpy as np
            from scripts.z13_eval import paired
            rng=np.random.default_rng(1);p=read(RUN/'protocol.json');pairs=[('D_masa','A_masa'),('D_masa','R1_masa'),('D_masa','legacyD'),('D_masa','C_masa'),('C_masa','A_masa'),('C_masa','R1_masa'),('C_masa','legacyC')]
            for left,right in pairs:
                obj=paired(report['per_video'],left,right,p['videos'],rng);saved=report['diff'][left+'-'+right+'|all'];assert all(saved[k]==v for k,v in obj.items())
            passed=all(report['criteria'].values());uncertain=any(report['diff'][key+'|all']['ci95'][0]<=0<=report['diff'][key+'|all']['ci95'][1] for key in ('D_masa-A_masa','D_masa-R1_masa','D_masa-legacyD'))
            state='completed_positive' if passed else ('inconclusive' if uncertain else 'completed_negative');card.update(state=state,done=f'{now():%Y-%m-%d}')
            obtained='runs/z13_center_hota_20261021/eval/report.json → diff: '+str(report['diff'])+'; criteria '+str(report['criteria'])+'. Все35видео и исходные HOTA стенда/legacyC/legacyD воспроизведены, основные парные интервалы пересчитаны.'
            interpretation='Основные критерии полного стенда '+('выполнены.' if passed else 'не выполнены.')+' Это одна диагностическая политика; результаты C и D−C не подменяют исход D.'
            decision='Сохранить фиксированную версию и отдельно согласовать последующую независимую проверку.' if passed else 'Эту политику не считать подтверждённым улучшением стенда; порог по dev/HOTA не подбирать. Следующая замена — отдельная проверяемая постановка.'
            for i,k in enumerate(('implementation_D_vs_stand','fly_added_D_vs_R1','readout_D_vs_legacyD'),1):card['acceptance'][i].update(met=bool(report['criteria'][k]),evidence='runs/z13_center_hota_20261021/eval/report.json')
            base=report['pooled']['A_masa|all']['HOTA'];gain=report['diff']['D_masa-A_masa|all']['pooled_gain'];tail='Относительно стенда: HOTA '+f'{100*gain/base:+.5f}%'.replace('.',',')+'. Доля измеренного потолка не определена: сопоставимый оракул этой замкнутой политики не измерялся.'
        else:
            card.update(state='ready',done='');obtained='runs/z13_center_hota_20261021/full_controller_result.json: завершено '+str(result['completed_jobs'])+'/'+str(result['total_jobs'])+' заданий; осталось '+str(len(result['remaining']))+'. all_own_workers_ended=true; незавершённое не помечено готовым. Полной HOTA нет.'
            interpretation='Очередь остановлена/завершила окно до полного35-видео сравнения. Это сохранение остатка, не вердикт об эффективности.'
            decision='Возобновить с готовых completed_jobs после возвращения автора; научные источники, модели и1%политика прежние. Ни одного завершённого задания не повторять без проверки повреждения.'
            tail='HOTA, процент прироста и доля потолка не определены: полного сопоставимого измерения нет.'
        for field,value in [('summary',interpretation),('obtained',obtained),('interpretation',interpretation),('decision',decision)]:revise(card,field,value,head,'Проверенное окончание полного или ограниченного утреннего окна.')
        for name in ('full_controller_result.json','jobs.json','execution_identity.json'):card['evidence'].append(dict(kind='file',ref='runs/z13_center_hota_20261021/'+name,claim='Полнота/остаток и конец собственного дерева'))
        journal('итог или сохранение очереди центрированного трекера','*Получено.* '+obtained+'\n\n*Вывод.* '+interpretation+'\n\n*Решение.* '+decision+'\n\n'+tail+' Формулировки проверены на предмет/инструмент, повторы и соответствие реально выполненному.')
        write(RUN/'night_finished.json',dict(own_workers_ended=True,complete=complete,recorded=now().isoformat()))
    write(path,cat);write(marker,dict(recorded=now().isoformat(),phase=x.phase,commit_before=head));print(x.phase,'recorded')


if __name__=='__main__':main()
