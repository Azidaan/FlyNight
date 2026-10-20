"""Record observed-memory progress/results without touching frozen science."""
from __future__ import annotations
import argparse
from datetime import datetime,timedelta,timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.z13_observed_memory_20261021 import RUN,IDENT,read,write,sha,verify,journal
MSK=timezone(timedelta(hours=3))


def update(cat,card,field,value,commit,reason):
    old=card.get(field,'')
    if old and old!=value:
        card.setdefault('revisions',[]).append(dict(date=f'{datetime.now(MSK):%Y-%m-%d}',commit=commit,field=field,old=old,reason=reason))
    card[field]=value


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--running',action='store_true');ap.add_argument('--finish',action='store_true');a=ap.parse_args()
    catpath=ROOT/'experiments/project_history.json';cat=read(catpath);card=next(c for c in cat['experiments'] if c['id']==IDENT)
    commit=subprocess.check_output(['git','--no-optional-locks','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    p=read(RUN/'protocol.json');verify(p)
    if a.running:
        marker=RUN/'running_recorded.json'
        if marker.exists():return
        card['state']='running'
        update(cat,card,'summary','Выполняется ограниченный функциональный пилот C/D/R1. Восстановление3299 эпизодов на46 видео точно совпало с кэшем; проба стоимости и воспроизведения прошла. Научного итога ещё нет.',commit,'Запущен основной пилот после успешной технической подготовки; прежняя постановка сохранена.')
        for name,claim in [('replay_report.json','Точные наблюдаемые рамки восстановлены без разметки и видео'),('benchmark.json','Стоимость и воспроизведение прежнего D на фиксированной пробе'),('recovery1_protocol.json','Служебное восстановление атомарной записи без изменения науки и исходного окна')]:
            card['evidence'].append(dict(kind='file',ref=(RUN/name).relative_to(ROOT).as_posix(),claim=claim))
        write(catpath,cat)
        journal('основной пилот памяти и служебное восстановление','*Получено.* `runs/z13_observed_memory_20261021/replay_report.json`:46 видео,3299 эпизодов; все кадры/вырезки и последние маски совпали, без разметки и декодирования видео. `benchmark.json`: оценка с запасом892,89с; контроль D на68 строках воспроизведён: центр<0,000031px, балл<0,0000047, lambda совпадает.\n\n*Действие.* Один последний образец заменён банком различных наблюдений до потери; основное сравнение заранее задано для C, D — отдельный контроль. Полный этап запущен в01:44:47 МСК с бюджетом1200с+15с. Первый worker завершился с exit1 при временном WinError5 обновления status.json; ошибка и исходные источники сохранены. Служебная версия v2 повторяет только атомарную запись, использует общий GPU-мьютекс и новые логи; два теста этой записи пройдены. Возобновление не продлевает исходное окно: `controller.json` — окончание дерева до02:04:59 МСК, раньше исходного02:05:02. Консоль переведена на PowerShell7 с явным чтением дат как строк. Научные настройки, данные, ядра, критерии и допуски не изменялись. HOTA не измерялась.')
        write(marker,dict(card=IDENT,previous_commit=commit,recorded_at=datetime.now(MSK).isoformat()))
        return
    if not a.finish:raise ValueError('Choose --running or --finish')
    marker=RUN/'full_finish_recorded.json'
    if marker.exists():raise RuntimeError('Full result already recorded')
    result=read(RUN/'full_recovery1_controller_result.json')
    if not result['worker_ended'] or result['exit_code']!=0 or result['hard_timeout']:raise RuntimeError('No successful full result; partial outcome must be recorded separately')
    import numpy as np
    from scripts.z13_continue import paired,per_video_useful
    replay=read(RUN/'replay_report.json')
    for record in replay['records']:
        assert sha(RUN/'observations'/(record['video']+'.npz'))==record['output_sha256']
    report=read(RUN/'report.json');pv={};reproductions={}
    for arm in ('C','D','R1'):
        assert (RUN/arm/'done.json').exists()
        reproduction=read(RUN/arm/'reproduction.json')
        assert reproduction['max_center_px']<.5 and reproduction['max_peak_score']<.001 and reproduction['lambda_matches']
        reproductions[arm]=reproduction
        for mode in ('single','memory'):
            name=arm+'_'+mode;rows=read(RUN/arm/(mode+'_rows.json'))
            for r in rows:
                if r['kind']==1:assert r['useful']==(r['err']<=.25*r['scale'] and r['err']<r['err_pred'])
            pv[name]=per_video_useful(rows)
            assert np.isclose(np.mean(list(pv[name].values())),report['summary'][name]['useful_mean_video'],rtol=0,atol=1e-14)
    pairs=[('C_memory','C_single'),('C_memory','R1_memory'),('D_memory','D_single'),('D_memory','C_memory'),('D_memory','R1_memory')]
    for x,y in pairs:assert paired(pv[x],pv[y])==report['paired'][x+'-'+y]
    audit=dict(all_done=True,worker_ended=True,exit_code=0,rows_and_paired_metrics_recomputed=True,source_hashes_unchanged=True,observed_masks_unchanged=True,reproductions=reproductions,report_sha256=sha(RUN/'report.json'))
    write(RUN/'result_audit.json',audit)
    c=report['paired']['C_memory-C_single'];eye=report['paired']['C_memory-R1_memory'];d=report['paired']['D_memory-D_single']
    assert report['primary_pass']==(c['ci95'][0]>0 and eye['ci95'][0]>0)
    state='completed_positive' if report['primary_pass'] else ('completed_negative' if c['ci95'][1]<0 else 'inconclusive')
    pct=lambda v:f'{100*v:.3f}'.replace('.',',')
    interval=lambda obj:f"{pct(obj['mean_per_video'])} п.п. [{pct(obj['ci95'][0])}; {pct(obj['ci95'][1])}]"
    base=report['summary']['C_single']['useful_mean_video'];new=report['summary']['C_memory']['useful_mean_video']
    rel=(new/base-1)*100 if base else None
    rel_text='не определено' if rel is None else f'{rel:+.3f}%'.replace('.',',')
    obtained=f"runs/z13_observed_memory_20261021/report.json → paired: C память−C один образец {interval(c)}; C память−R1 память {interval(eye)}; D память−D один образец {interval(d)}. summary: C {pct(base)}%→{pct(new)}%, относительное изменение {rel_text} при исходном считывателе как базе. full_recovery1_controller_result.json: worker_ended=true,exit0; result_audit.json: воспроизведение и пересчёт проверены."
    interpretation=('Ограниченная память отдельных наблюдений улучшила считывание C и превзошла тот же узел на входе глаза в принятых условиях.' if report['primary_pass'] else 'Основной критерий этой реализации памяти не выполнен. Это не устанавливает причину результата и не закрывает направление памяти наблюдений.')
    interpretation+=' Выборка train26 использовалась многократно; интервалы поисковые. Это функциональное сравнение фиксированных ядер и способов считывания, а не измерение HOTA.'
    decision=('Сохранить как функционального кандидата; следующая ступень требует заранее фиксированного причинного правила применения и отдельной проверки пользы стенду.' if report['primary_pass'] else 'Не переносить эту реализацию в стенд и не подбирать её настройки по train26. Альтернативы для согласования: память отдельных кандидатов по KeepTrack либо чтение наблюдений по текущему запросу, а не одна совместная гребневая модель. Узел причины провала пока не установлен.')
    card['state']=state;card['done']=f'{datetime.now(MSK):%Y-%m-%d}'
    reason='Полный проверенный результат; прежнее состояние и постановка сохраняются.'
    for field,value in [('summary',interpretation),('obtained',obtained),('interpretation',interpretation),('decision',decision),('cannot_claim','HOTA не измерялась. Не доказаны безопасный отказ, преимущество обучения D, анатомическая причина эффекта или бесполезность других способов памяти. Банк наблюдений не является полной реализацией STMTrack.')]:update(cat,card,field,value,commit,reason)
    for i,(met,ev) in enumerate([(True,'result_audit.json'),(c['ci95'][0]>0,'report.json → paired.C_memory-C_single'),(eye['ci95'][0]>0,'report.json → paired.C_memory-R1_memory')]):card['acceptance'][i].update(met=bool(met),evidence='runs/z13_observed_memory_20261021/'+ev)
    for name in ('report.json','result_audit.json','full_recovery1_controller_result.json'):
        card['evidence'].append(dict(kind='file',ref=(RUN/name).relative_to(ROOT).as_posix(),claim='Первичный полный результат или проверка его целостности'))
    card['confirmed_results']=[]
    for arm,obj in [('C',c),('D',d)]:
        if obj['ci95'][0]>0:
            src=(RUN/'report.json').relative_to(ROOT).as_posix();digest=sha(RUN/'report.json')
            card['confirmed_results'].append(dict(id=arm+'_memory_vs_single_train26',title=f'Память наблюдений против одного образца — {arm}',category='readout',metric='useful_center_fraction',unit='fraction',
                evidence=dict(source=src,locator=['paired',arm+'_memory-'+arm+'_single'],sha256=digest),
                baseline=dict(source=src,locator=['summary',arm+'_single','useful_mean_video'],sha256=digest),
                subset='train26,26 видео; повторно использованная выборка',comparison=f'{arm}: отдельные наблюдения до потери против последнего образца.',
                conditions='Те же последовательности признаков, фиксированное ядро, подбор lambda только на прежних20 train, парный бутстрэп по видео.',limitations='Поисковый функциональный результат, не HOTA и не доказательство безопасного применения. Частный эффект не заменяет общий критерий карточки.'))
    write(catpath,cat)
    journal('полный результат памяти наблюдений','*Получено.* '+obtained+'\n\n*Вывод.* '+interpretation+'\n\n*Решение.* '+decision+'\n\nСводка к исходному варианту: C, доля полезных центров, '+rel_text+'; доля потолка не определена, соответствующий потолок не измерялся. HOTA не измерялась. Научные формулировки проверены на различие предмета/инструмента, повторение и соответствие фактически выполненному.')
    write(marker,dict(card=IDENT,state=state,primary_pass=report['primary_pass'],recorded_at=datetime.now(MSK).isoformat(),report_sha256=sha(RUN/'report.json')))
    print(json.dumps(dict(state=state,primary=report['primary_pass'],C_memory_effect=c,D_memory_effect=d),ensure_ascii=False))

if __name__=='__main__':main()
