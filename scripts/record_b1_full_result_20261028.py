"""Записать только полный независимо проверенный результат Б1 и его ограничения."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
from datetime import datetime
import sys,time,subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261028 import RUN,service
from scripts.b123_safe_journal_20261027 import journal
read,write,sha,now=service.read,service.write,service.sha,service.now

def main():
    import numpy as np
    folder=RUN/'b1_result_record';source=RUN/'b1_tracker';audit_folder=RUN/'b1_tracker_audit'
    r=read(source/'report.json');a=read(source/'independent_audit.json')
    assert r['complete'] and a['passed'] and a['full_series'] and a['videos']==20
    for previous in [source,audit_folder]:
        c=read(previous/'controller_result.json')
        assert c['exit_code']==0 and c['all_owned_ended'] and not any(service.matching(k,v) for k,v in c['owned'].items())
    for path,digest in r['output_sha256'].items():assert sha(ROOT/path)==digest,path
    assert all(len(m)==20 for m in r['per_video'].values())
    base=r['pooled']['baseline']['HOTA'];d=r['pooled']['D']['HOTA'];appearance=r['pooled']['appearance']['HOTA']
    rng=np.random.default_rng(291028);videos=list(r['per_video']['baseline'])
    differences=np.array([r['per_video']['appearance'][v]['HOTA']-r['per_video']['baseline'][v]['HOTA'] for v in videos])
    boot=differences[rng.integers(len(videos),size=(5000,len(videos)))].mean(1)
    analysis=dict(appearance_vs_baseline=dict(mean_per_video=float(differences.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),n=20),
        exploratory_comparison_added_after_complete_run=True,pooled_D_relative_percent=100*(d-base)/base,
        source=(source/'report.json').relative_to(ROOT).as_posix(),source_sha256=sha(source/'report.json'),
        independent_audit_sha256=sha(source/'independent_audit.json'),updated=now())
    write(folder/'analysis.json',analysis)
    history=read(ROOT/'experiments/project_history.json');card=next(c for c in history['experiments'] if c['id']=='b1_full_tracker_pilot_v1')
    report_ref=(source/'report.json').relative_to(ROOT).as_posix();audit_ref=(source/'independent_audit.json').relative_to(ROOT).as_posix()
    main=bool(r['main_acceptance'])
    changes=dict(state='completed_positive' if main else 'inconclusive',done='2026-10-28',
        obtained=f'{report_ref}: complete=true/hota_measured=true, все20видео/7вариантов; pooled.HOTA baseline={base}, appearance={appearance}, D={d}. paired.D-baseline={r["paired"]["D-baseline"]}; paired.D-appearance={r["paired"]["D-appearance"]}. {audit_ref}: passed=true/full_series=true/videos=20, все назначения/головы/raw/JSON/метрики повторены; ранее завершённый аудит первого видео переиспользован поSHA. Оба контроллера code0/all_owned_ended; готовые19GPUне пересчитывались.',
        interpretation=('Пилот прошёл все заранее заданные сравненияD; одинseed всё равно не подтверждает всюБ1.' if main else 'Полный вариантD лучше исходного трекера, но его дополнительное преимущество над той же ветвью внешнего вида, C, R1 и B не подтверждено. Основной критерий пилота не пройден. Общая прибавка к исходному трекеру не доказывает вклад ядраFlyVis; отдельно обученные альтернативы и исторические20ограничивают вывод.'),
        decision='Зафиксировать полный пилот с частными подтверждениями и непройденными критериями. Не менять веса/пороги по этим20и не повторять готовый прогон. Продолжить прямо порученную проверку представлений1/3/4 отдельно и совместно: реальные входы/общий обучаемый интерфейс/стоимость/полный трекер. Полное подтверждениеБ1по трём начальным случаям этим пилотом не заменяется.',
        cannot_claim='Одинseed26/одна эпоха и20многократно использованных val не новый независимый итоговый тест и не универсальная оценкаБ1–Б3. Прибавка всего вариантаDкnative не изолирует ядро. Интервал через0 не доказывает эквивалентности. Старые gatefull-box/categorical_cost и редкие train/live-запросы—ограничения постановки, не установленные причины. Прежние таймауты/поля сохранены вrevisions.')
    for field,value in changes.items():
        card.setdefault('revisions',[]).append(dict(date='2026-10-28',commit='',field=field,old=card.get(field,''),reason='Завершён полный20пилот и независимый аудит в новом разрешённом окне'))
        card[field]=value
    card['acceptance'][0].update(met=True,evidence=audit_ref+':passed/full_series/videos20, код0 и фактический конец обоих деревьев')
    card['acceptance'][1].update(met=main,evidence=report_ref+':main_acceptance='+str(main)+';paired Dкbaseline/appearance/C/R1/B; одинseedнеполнаяБ1')
    card.setdefault('revisions',[]).append(dict(date='2026-10-28',commit='',field='metrics',old=card.get('metrics',[]),reason='Числа одного видео заменены полным20итогом; прежние сохранены'))
    card['metrics']=[dict(key='HOTA',value=r['pooled'][arm]['HOTA'],baseline=base,subset='20историческихval;'+arm+';фиксированные весаseed26/однаэпоха',source=report_ref,locator=['pooled',arm,'HOTA']) for arm in ['baseline','appearance','C','D','R1','B']]
    card['evidence'] += [dict(kind='file',ref=report_ref,claim='Полные20, совокупные метрики и заранее заданные парные сравнения'),dict(kind='file',ref=audit_ref,claim='Полный независимый аудит и повтор метрик')]
    result=dict(id='b1_D_vs_native20',title='Полный вариант Б1 против исходного трекера — частная прибавка HOTA',category='tracker',metric='HOTA',unit='points',
        comparison='Весь вариантDсветвьювнешнего вида противnative XYWHByteTrack',subset='20историческихval',
        conditions='Те же детекции/YOLO/7вариантов, сохранённые весаоднойэпохиseed26, никакихпороговпоval; полный независимый аудит.',
        limitations='Это польза всей связки к исходной базе, не отдельный вклад ядраFlyVis. Общий критерий превосходстваDнадappearance/C/R1/Bне пройден. Одинseed и историческийval.',
        evidence=dict(source=report_ref,sha256=sha(source/'report.json'),locator=['paired','D-baseline']))
    if r['paired']['D-baseline']['ci95'][0]>0:card['confirmed_results']=[result]
    write(ROOT/'experiments/project_history.json',history)
    journal('b1_full_pilot_completed','Б1:полные20и независимый аудит завершены, дополнительная пользаDне подтверждена' if not main else 'Б1:полный пилот завершён',
        '*Получено.* `'+report_ref+'`:совокупнаяHOTA baseline='+str(base).replace('.',',')+', appearance='+str(appearance).replace('.',',')+', D='+str(d).replace('.',',')+'. '
        'Парные интервалы—вpaired,совокупнуюдельтуне смешивать со среднейпарной. `'+audit_ref+'`:passed/full_series/20. '
        '*Вывод.* '+changes['interpretation']+' *Решение.* '+changes['decision']+' ПроцентDксовокупнойбазе—`b1_result_record/analysis.json`; измеренного потолка для доли не задавали.')
    end=datetime.fromisoformat(read(folder/'execution_deadline.json')['deadline']).timestamp();checks=[]
    for i,args in enumerate([['scripts/validate_project_history.py'],['-m','pytest','tests/test_project_history.py','-q'],['scripts/restore_obsidian_sections_20261027.py'],['-m','pytest','tests/test_obsidian_fundamental_overview.py','-q']]):
        write(folder/'status.json',dict(state='running',step=i,command=args,updated=now()),required=False)
        with (folder/f'check_{i}.log').open('w',encoding='utf-8') as log:
            c=subprocess.run([sys.executable,*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,end-time.time()-10))
        checks.append(dict(command=args,exit_code=c.returncode));assert c.returncode==0,args
    write(folder/'report.json',dict(complete=True,main_acceptance=main,state=changes['state'],checks=checks,updated=now()))
    with (RUN/'HANDOFF.md').open('a',encoding='utf-8') as f:
        f.write('\n\nПолныйБ1иоформление025завершены: b1_tracker/report.json/independent_audit.json полные20, b1_result_record/report.jsonвсеchecks0. Общийmain_acceptance='+str(main)+'. ПодтверждениеDкnativeчастное,неизолируетядро. Дальше030источники/040экспорт и реализацияматрицы1/3/4;готовыеБ1неповторять.\n')

if __name__=='__main__':main()
