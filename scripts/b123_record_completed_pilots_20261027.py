"""Независимая сверка завершённых ступеней и запись, без повторения обучения/HOTA."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
from datetime import datetime,timezone,timedelta
import hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
RUN=ROOT/'runs/b123_20261027'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def ended(path):
    r=read(path);assert r['exit_code']==0 and r['all_owned_ended'] and not r['timed_out']
    import psutil
    for pid,birth in r['owned'].items():
        if psutil.pid_exists(int(pid)):assert abs(psutil.Process(int(pid)).create_time()-birth)>=.01

def main():
    assert not (RUN/'completed_pilots_independent_audit.json').exists()
    ended(RUN/'b1_training/recovery2_controller.json')
    for name in ('b2_gradient/recovery1','b3_gradient','busca_readiness','ltr_readiness','recovery_motion_control'):
        ended(RUN/name/'controller_result.json')
    b1=read(RUN/'b1_training/report.json');assert b1['complete'] and not b1['hota_measured']
    for arm,a in b1['arms'].items():
        assert a['complete'] and a['optimizer_steps']==10309 and a['evaluated_queries']==4183
        for name,key in (('checkpoint.pt','checkpoint_sha256'),('probability.npz','probability_sha256')):
            assert sha(RUN/'b1_training'/arm/name)==a[key]
        assert a['implementation']['zero_start_exact'] and a['implementation']['checkpoint_restore_exact']
    for contrast,c in b1['paired'].items():
        a,b=contrast.split('-');keys=sorted(b1['per_video'][a])
        dif=np.array([b1['per_video'][a][v]-b1['per_video'][b][v] for v in keys])
        assert abs(float(dif.mean())-c['mean_per_video'])<1e-15
        assert c['ci95'][0]<=0<=c['ci95'][1]
    motion=read(RUN/'recovery_motion_control/report.json');assert motion['complete']
    old=ROOT/'runs/z13_val20_matched_20261024/raw/xywh/raw';output_hashes={}
    for v in motion['videos']:
        assert read(RUN/'recovery_motion_control/baseline'/(v+'.json'))==read(old/(v+'.json'))
        base=read(RUN/'recovery_motion_control/baseline'/(v+'.json'))
        ext=read(RUN/'recovery_motion_control/emit_lost_kalman'/(v+'.json'))
        assert len(base)==len(ext)
        for a,b in zip(base,ext):
            assert b['labels'][:len(a['labels'])]==a['labels']
            assert len({str(l['id']) for l in b['labels']})==len(b['labels'])
        for arm in ('baseline','emit_lost_kalman'):
            path=RUN/'recovery_motion_control'/arm/(v+'.json');output_hashes[path.relative_to(ROOT).as_posix()]=sha(path)
    keys=sorted(motion['videos']);delta=np.array([motion['per_video']['emit_lost_kalman'][v]['HOTA']-motion['per_video']['baseline'][v]['HOTA'] for v in keys])
    assert (delta<0).all() and abs(float(delta.mean())-motion['paired']['mean_per_video'])<1e-12
    # Иная организация бутстрэпа и другой фиксированный сид, только проверка знака.
    generator=np.random.default_rng(2701)
    samples=[float(np.mean(generator.choice(delta,size=len(delta),replace=True))) for _ in range(5000)]
    independent_ci=np.quantile(samples,[.025,.975]).tolist();assert independent_ci[1]<0
    b2=read(RUN/'b2_gradient/recovery1/report.json');b3=read(RUN/'b3_gradient/report.json')
    assert b2['passed'] and b2['detector_weights_unchanged'] and b3['passed']
    assert all(a['zero_detector_exact'] and a['steps']==3 for a in b2['arms'].values())
    assert all(a['zero_start_exact'] and a['steps']==3 for a in b3['arms'].values())
    source='runs/b123_20261027/recovery_motion_control/report.json'
    audit=dict(passed=True,at=datetime.now(timezone(timedelta(hours=3))).isoformat(),
        b1_probability_and_checkpoint_hashes_verified=True,b1_saved_action_contrast_means_verified=True,
        b1_all_contrasts_intervals_include_zero=True,
        motion_baseline_json_semantically_exact_all20=True,motion_only_appends_no_duplicate_ids=True,
        motion_all20_HOTA_differences_negative=True,independent_seed2701_ci95=independent_ci,
        output_sha256=output_hashes,b2_service_passed_original_deadline=True,b3_implementation_passed=True)
    write(RUN/'completed_pilots_independent_audit.json',audit)
    path=ROOT/'experiments/project_history.json';raw=path.read_bytes();cat=json.loads(raw)
    date='2026-10-27'
    def update(ident,state,fields,evidence,acceptance=None):
        c=next(x for x in cat['experiments'] if x['id']==ident)
        for key,value in fields.items():
            previous=c.get(key,'')
            if previous and previous!=value:
                c.setdefault('revisions',[]).append(dict(date=date,commit='',field=key,old=previous,reason='Первичный итог завершённой ступени;публикация заблокирована прежним index.lock.'))
            c[key]=value
        c.update(state=state,done=date)
        c.setdefault('evidence',[]).append(dict(kind='file',ref=evidence,claim='Первичный завершённый отчёт'))
        if acceptance is not None:
            for item,met in zip(c['acceptance'],acceptance):item.update(met=met,evidence=evidence)
    update('b1_temporal_residual_training_v1','inconclusive',dict(
        summary='Первая эпоха и все контроли завершены;преимущество по сохранённым действиям не подтверждено,полная HOTA не измерена.',
        obtained='runs/b123_20261027/b1_training/report.json:arms — C/D/R1/B по10309шагов и4183оценённыхзапроса;summary.D.counts — 93выбора,45полезных/24вредных;paired.D-base.mean_per_video=0,0007706644382168869,ci95=[−0,0019298403601411746;0,0036207225691956644]. Все4интервала D-контроль включают0. Данные/хеши сверены completed_pilots_independent_audit.json.',
        interpretation='Вычислительный путь работает,ядроDобновилось. Однаэпоха не подтвердила преимущество по сохранённым действиям. Это не отрицательный HOTA и не закрытие Б1.',
        decision='Сохранить веса и неподтверждённый пилот;по прямому поручению автора приоритет ночи передан BUSCA/LTR. Полный рантаймБ1 и HOTA остаются незавершёнными;готовое обучение не повторять.'),
        'runs/b123_20261027/b1_training/report.json',[True,True,True])
    for ident,folder,title in [('b2_predetection_gradient_v1','b2_gradient/recovery1','Временная ветвь до обнаружения'),('b3_object_memory_gradient_v1','b3_gradient','Память объектов')]:
        r=read(RUN/folder/'report.json')
        update(ident,'infrastructure',dict(summary=title+':проверка вычислительного пути прошла;научная гипотеза не проверена.',
            obtained=(RUN/folder/'report.json').relative_to(ROOT).as_posix()+': '+json.dumps(r,ensure_ascii=False),
            interpretation='Конечные потери,нулевое сохранение базы и обучаемые производные подтверждены на одном клипе;качество/сходимость/HOTAнеизмерены.',
            decision='Сохранить техническую готовность;не раскрашивать главную гипотезу как проверенную и не продвигать пробные веса. Б2:исходнаяошибка/recovery1с прежнимсрокомсохранены.' if ident.startswith('b2') else 'Сохранить техническую готовность;не раскрашивать главную гипотезу как проверенную и не продвигать пробные веса.'),
            (RUN/folder/'report.json').relative_to(ROOT).as_posix(),[True,True,True])
    for method in ('busca','ltr'):
        r=read(RUN/(method+'_readiness/report.json'))
        update(method+'_recovery_readiness_v1','infrastructure',dict(
            summary='Подготовительная проверка завершена;полный запуск метода пока не готов.',
            obtained=(RUN/(method+'_readiness/report.json')).relative_to(ROOT).as_posix()+': '+json.dumps(r,ensure_ascii=False),
            interpretation='Найдены ограничения готовности к воспроизведению;это не отрицательный результат метода.',
            decision=r['next_action']), (RUN/(method+'_readiness/report.json')).relative_to(ROOT).as_posix(),[True if method=='busca' else False,True,True])
    base=motion['pooled']['baseline'];ext=motion['pooled']['emit_lost_kalman']
    difference=ext['HOTA']-base['HOTA'];relative=100*difference/base['HOTA']
    update(IDENT if False else 'recovery_motion_output_control_v1','completed_negative',dict(
        summary='Простая выдача калмановских прогнозов ухудшила HOTA на всех20видео;нужна проверка отказа от ложного продолжения.',
        obtained=source+' → pooled: '+json.dumps(motion['pooled'],ensure_ascii=False)+';paired: '+json.dumps(motion['paired'],ensure_ascii=False)+'. Независимаяпроверка:completed_pilots_independent_audit.json.',
        interpretation='Рамки существующего выхода и ассоциация не менялись. Простое продолжение уменьшило пропуски,но породило существенно больше ложных рамок;это отрицательный результат этой выдачи,не BUSCA/LTR. Главная мишень обучаемого модуля—надёжный отказ от продолжения.',
        decision='Не переносить простую выдачу в стенд. Сохранить её как контроль для BUSCA/масок. Проверять,отсекает ли обучаемое решение ложные продолжения при сохранении полезных.'),source,[True,True,False])
    c=next(x for x in cat['experiments'] if x['id']=='recovery_motion_output_control_v1')
    c['metrics']=[dict(key='HOTA',value=ext['HOTA'],baseline=base['HOTA'],subset='same20val raw XYWH+output-only lostKalman,noMASA',source=source,locator='pooled.emit_lost_kalman.HOTA')]
    assert path.read_bytes()==raw;write(path,cat)
    from scripts import z13_localization_labels_20261023 as J
    J.IDENT='b123_completed_and_recovery_control'
    J.journal('Б1:пилотбезподтверждённойприбавки;Б2/Б3:техническийпуть;контрольвосстановленияотрицателен',
        '*Получено.* Всеисточники/сверка: `runs/b123_20261027/completed_pilots_independent_audit.json`. '
        'Б1 `b1_training/report.json`:C/D/R1/Bзавершены,всеинтервалы D-контрольвключают0,полнойHOTAнет. '
        'Б2послеотдельногоисправленияget_cfgуспешенвпрежнемсроке: `b2_gradient/recovery1/report.json`;Б3 `b3_gradient/report.json` passed=true. Обапроверяюттолькопутьпроизводныхнаодномклипе,неновизну/качествогипотез.\n\n'
        '*Контрольвосстановления.* `'+source+'`:тотже20valбезMASA,исходныйвыходпобайтнопочисламвоспроизведён. HOTA '
        f'{base["HOTA"]:.6f}→{ext["HOTA"]:.6f},общаяразность{difference:.6f}п.п.;средняяповидео{motion["paired"]["mean_per_video"]:.6f},ДИ{motion["paired"]["ci95"]}. '
        f'CLR_FN{base["CLR_FN"]}→{ext["CLR_FN"]},CLR_FP{base["CLR_FP"]}→{ext["CLR_FP"]}. '
        'Вреднавсех20. Этоизмереннаяпростаявыдачапрогноза,непрогонBUSCA/LTRинепотолок.\n\n'
        '*Решение.* Непродвигатьпростуювыдачу. Следующиймодульдолженпроверятьотказотложногопродолжения. '
        'ПродолжениеBUSCA→LTRразрешеноавторомдо07:30,веса/совместимостьещёнужны. Автоматизацияразв30минутдлянеобходимойреализации,локальнаяочередьдляготовыхпереходов. '
        f'Сводкакбазе:контрольвосстановления→HOTA→{relative:.4f}%кбазе;доляпотолканеизвестна. ДляБ1–Б3HOTAнеизмерена,процентневычисляется. ГрафГипотезынеменяется.')
    with (RUN/'HANDOFF.md').open('a',encoding='utf-8') as f:
        f.write('\n\n## '+datetime.now(timezone(timedelta(hours=3))).strftime('%H:%M')+' МСК — завершённые пилоты и отрицательный контроль\n\n'
            'Читай completed_pilots_independent_audit.json и первичные report. Б1все4контроляготовы,всеДИпоsavedactionсодержат0;HOTAБ1нет. '
            'Б2послеслужебногоget_cfg(overrides=args)прошластарыйсрок,Б3техническипрошла,весаscratchнепродвигать. '
            'Общийконтрольrecovery_motion_control:HOTA57,900829→49,329032;вред20/20,интервалполностью<0. '
            'FNменьше,FPзначительнобольше. НеназыватьэтопроваломBUSCA/LTR. Цельпродолжения—обучаемыйотказотложныхпродолжений. '
            'BUSCA/LTRещёнеисполнялись;ищипроверяемыевеса/точнуюреализацию,готовьрабочиймодуль. Гипотезызаморожены. '
            'Нужныпроверкикаталогапослеэтойзаписи,защищённаяпересборкаобычногографа. .git/index.lockнетронут.\n')
    print(json.dumps(dict(recorded=True,motion_HOTA_change=difference,motion_relative_percent=relative)))

if __name__=='__main__':main()
