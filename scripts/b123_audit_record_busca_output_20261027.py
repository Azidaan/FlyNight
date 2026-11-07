"""Независимо проверить сохранённые результаты BUSCA и записать отрицательный итог."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
from datetime import datetime,timezone,timedelta
import hashlib,json,sys
import numpy as np
import psutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
RUN=ROOT/'runs/b123_20261027/busca_output_pilot'
SOURCE='runs/b123_20261027/busca_output_pilot/report.json'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert not (RUN/'independent_audit.json').exists()
    end=read(RUN/'controller_result.json');assert end['exit_code']==0 and end['all_owned_ended'] and not end['timed_out']
    for pid,birth in end['owned'].items():
        if psutil.pid_exists(int(pid)):assert abs(psutil.Process(int(pid)).create_time()-birth)>.01
    r=read(RUN/'report.json');p=read(RUN/'protocol.json')
    assert r['complete'] and len(r['videos'])==20 and set(r['videos'])==set(p['videos'])
    assert all(r['baseline_arrays_exact'].values()) and not r['tracker_state_changed'] and not r['original_busca_reproduced']
    for name,h in p['sha256'].items():assert sha(ROOT/name)==h,name
    control=read(ROOT/'runs/b123_20261027/recovery_motion_control/report.json');hashes={};added=0
    for video in r['videos']:
        base=read(RUN/'baseline'/(video+'.json'))
        original=read(ROOT/'runs/z13_val20_matched_20261024/raw/xywh/raw'/(video+'.json'))
        recovered=read(RUN/'busca_output'/(video+'.json'))
        assert base==original and len(base)==len(recovered)
        assert r['per_video']['baseline'][video]==control['per_video']['baseline'][video]
        for a,b in zip(base,recovered):
            assert a['name']==b['name'] and a['videoName']==b['videoName']
            assert b['labels'][:len(a['labels'])]==a['labels']
            assert len({str(x['id']) for x in b['labels']})==len(b['labels'])
            added+=len(b['labels'])-len(a['labels'])
        for arm in ('baseline','busca_output'):
            path=RUN/arm/(video+'.json');hashes[path.relative_to(ROOT).as_posix()]=sha(path)
    delta=np.array([r['per_video']['busca_output'][v]['HOTA']-r['per_video']['baseline'][v]['HOTA'] for v in sorted(r['videos'])])
    assert abs(float(delta.mean())-r['paired']['vs_baseline']['mean_per_video'])<1e-12
    assert int((delta>0).sum())==r['paired']['vs_baseline']['better']
    # Независимый сид и организация выборки: проверка знака, не замена первичного ДИ.
    rng=np.random.default_rng(27053)
    samples=np.array([rng.choice(delta,size=20,replace=True).mean() for _ in range(5000)])
    ci=np.quantile(samples,[.025,.975]).tolist();assert ci[1]<0
    simple=np.array([r['per_video']['busca_output'][v]['HOTA']-control['per_video']['emit_lost_kalman'][v]['HOTA'] for v in sorted(r['videos'])])
    assert (simple>0).all() and abs(float(simple.mean())-r['paired']['vs_simple_continuation']['mean_per_video'])<1e-12
    assert r['pooled']['baseline']==control['pooled']['baseline']
    result=dict(passed=True,at=datetime.now(timezone(timedelta(hours=3))).isoformat(),
        all20_baseline_json_exact=True,all20_baseline_metrics_exact=True,only_appended_unique_ids=True,
        frozen_sources_and_inputs_unchanged=True,owned_processes_ended=True,
        independently_recomputed_mean=float(delta.mean()),independent_ci95=ci,
        learned_recovery_beats_unconditional_all20=True,added_output_labels=added,output_sha256=hashes,
        scope='Проверка сохранённых выходов и статистики, без повторного обучения, инференса или TrackEval. Все промежуточные raw-массивы нового запуска не сохранялись; их точность проверил сам замороженный исполнитель до записи каждого видео.')
    (RUN/'independent_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    base=r['pooled']['baseline'];new=r['pooled']['busca_output'];difference=new['HOTA']-base['HOTA'];relative=100*difference/base['HOTA']
    path=ROOT/'experiments/project_history.json';before=path.read_bytes();catalog=json.loads(before)
    card=next(c for c in catalog['experiments'] if c['id']=='busca_output_recovery_pilot_v1')
    fields=dict(summary='Выходная адаптация BUSCA лучше безусловного продолжения, но ухудшает исходный XYWH. В стенд не принимается.',
        obtained=f'{SOURCE}: pooled.baseline.HOTA={base["HOTA"]}; pooled.busca_output.HOTA={new["HOTA"]}; paired.vs_baseline={r["paired"]["vs_baseline"]}. '
            f'CLR_FP {base["CLR_FP"]}→{new["CLR_FP"]}, CLR_FN {base["CLR_FN"]}→{new["CLR_FN"]}, IDSw {base["IDSw"]}→{new["IDSw"]}; источник pooled. '
            f'По сравнению с простым продолжением paired.vs_simple_continuation={r["paired"]["vs_simple_continuation"]}. '
            'Независимая сверка: runs/b123_20261027/busca_output_pilot/independent_audit.json.',
        interpretation='Обученный отказ отсеивает часть вреда безусловной выдачи, но оставшиеся ложные рамки перевешивают восстановленные пропуски. '
            'Это отрицательный результат конкретной выходной адаптации без изменения состояния, компенсации камеры, полного авторского отбора памяти и дообучения на BDD100K. Не опровержение полной BUSCA или всех временных ветвей.',
        decision='Не принимать эту выходную адаптацию в стенд. Сохранить её как контроль обучаемого отказа. '
            'Следующая разрешённая ветвь — восстановление по маскам; точную LTR не подменять другим методом. '
            'Возможные замены узла отказа: штатная полная BUSCA с её памятью и состоянием; масочная LTR с отдельным выбором правильной маски. Их пользу в нашем стенде ещё нужно измерить.',
        cannot_claim=p['limitations']+' '+p['adaptation']+' Не измерен потолок HOTA и не проверены научные Б1–Б3.')
    for key,value in fields.items():
        old=card.get(key,'')
        if old and old!=value:card.setdefault('revisions',[]).append(dict(date='2026-10-27',commit='',field=key,old=old,reason='Записан завершённый первичный итог, прежняя формулировка сохранена.'))
        card[key]=value
    card.update(state='completed_negative',done='2026-10-27')
    for entry,met in zip(card['acceptance'],[True,True,False]):entry.update(met=met,evidence=SOURCE)
    card['metrics']=[dict(key='HOTA',value=new['HOTA'],baseline=base['HOTA'],subset='same20val rawXYWH;output-onlyBUSCA;noMASA',source=SOURCE,locator='pooled.busca_output.HOTA')]
    card['evidence'] += [dict(kind='file',ref=SOURCE,claim='Полный отрицательный итог20видео'),
        dict(kind='file',ref='runs/b123_20261027/busca_output_pilot/independent_audit.json',claim='Независимая сверка сохранённых выходов и знака интервала')]
    assert path.read_bytes()==before;path.write_text(json.dumps(catalog,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    from scripts import z13_localization_labels_20261023 as J
    J.IDENT='busca_output_recovery_pilot_v1'
    J.journal('выходная адаптация BUSCA: вред меньше простого продолжения, но HOTA ниже базы',
        '*Получено.* '+fields['obtained']+'\n\n*Вывод.* '+fields['interpretation']+'\n\n*Решение.* '+fields['decision']+'\n\n'
        f'*Сводка к базе.* Адаптация BUSCA → HOTA → {relative:.6f}% к базе, общая разность {difference:.6f}п.п. '
        'Доля измеренного потолка неизвестна. Полная серия заняла863,031с; источник report.json.seconds. '
        'Код0, all_owned_ended=true, PID/создание проверены. Готовые видео и TrackEval не повторялись.')
    with (ROOT/'runs/b123_20261027/HANDOFF.md').open('a',encoding='utf-8') as f:
        f.write('\n\n## '+J.now().strftime('%H:%M')+' МСК — полный итог выходной BUSCA отрицателен\n\n'
            'busca_output_pilot/report.json complete=true, controller_result код0/всёдеревозавершено. '
            'independent_audit.json проверяет все20baselineJSON,метрики,неизменныеисточники,append-only/uniqueIDs,знакДИ другимсидом. '
            'HOTA57.900829→56.843638;парная средняя−0.966471п.п.[−1.322852;−0.548141],лучше3/20. '
            'FP3331→6304,FN13525→12488,IDSw1001→973. К простому продолжению+7.303569п.п.[6.115147;8.544724],лучше20/20. '
            'Это отрицательная выходная адаптация,НЕопровержение полнойBUSCA. В стенд не принимать,готовое не повторять. '
            'Карточка обновлена с revisions и HOTA/source/locator; нужны валидация/58тестов и фундаментальная задача4вграфе. '
            'Далее масочнаяветвь по поручению. ExactLTRкод/селекторвеса пока не подтверждены,SGARнеподменять. '
            '.git/index.lock не трогать, одинGPU,07:20сохранение/07:29:30стоп.\n')
    print(json.dumps(dict(passed=True,HOTA=new['HOTA'],difference=difference,relative_percent=relative,paired=r['paired']['vs_baseline']),ensure_ascii=False))
if __name__=='__main__':main()
