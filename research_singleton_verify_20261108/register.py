"""Preregister own-template verification, keeping the first problem isolated."""
import sys,subprocess,time,ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_singleton_verify_20261108.common import *


def main():
    import psutil
    assert datetime.now(MSK)<datetime(2026,11,8,13,55,tzinfo=MSK)
    assert read(PREVIOUS/'recovery3_result.json')['passed'] and not read(PREVIOUS/'comparison_report.json')['full_tracker_gate']['proceed']
    assert read(RUN/'inventory_recovery1.json')['passed']
    protocol=dict(id='singleton_own_template_v1_20261108',registered=datetime.now(MSK).isoformat(),active_problem=1,deferred=[2,3,4],deadline='2026-11-08T14:00:00+03:00',
        hypothesis='An absolute comparison to the track own past template can veto a foreign sole candidate when no second rival track exists. Useful-system benefit and incremental FlyVis benefit are separate questions.',
        action='Keep all native first-stage matches fixed. On the same free sole-geometrically-valid Lost edges with C<1, subtract fixedalpha0.2 only if the own-template verification passes. No direct Kalman replacement, ownership transfer, deduplication change, input expansion or kernel modification.',
        alpha=.2,alpha_source='Previously selected only onfit20: runs/singleton_free_20261108/selection.json. Not retuned on26or20val.',
        spatial='Same cached feature-independent minimum-residual transform and original crop. Keep shared box cells; average repeated current samples per unique reference cell, then equal weight per reference cell. Same DIS valid-domain mask for all representations, at least3unique reference cells. Undefined scores abstain. Centered spatial correlation forIMAGE,R1,FlyVis; scale-corrected uncentered flow cosine forDIS.',
        memory='Last4actual tracker updates, padded as before; those boxes are Kalman-filtered observations, not raw detector boxes. Current sequence is the same causal prefix plus predicted search views. No future frame.',
        calibration=dict(cohort='Onlyfit20primary-known query rows with exactly one geometrically valid candidate and C<1, including occupied/already-owned examples as identity-verification examples only.',
            labels='Candidate detection identity versus historical target_identity from existing supervision. Unknowns excluded from fitting and explicitly counted, never negatives. NoGT in runtime.',
            thresholds=['off',-1,-.75,-.5,-.25,0,.25,.5,.75,1],
            objective='Maximize TPR-FPR (sensitivity minus false-positive rate). Tie: fewer false positives, more true positives, then larger threshold;off is the most conservative tie.',
            same_budget='IMAGE,R1,DIS,FlyVis,spatially permutedFlyVis and deterministic hash score use the identical threshold grid.',
            system_selection='Choose one ofIMAGE,R1,DIS,FlyVis by fit objective, fewer false positives, more true positives, then fixed simplicity orderIMAGE,R1,DIS,FlyVis. Freeze before any evaluate26 labels for this comparison.'),
        controls=dict(B0='Unchanged native output, reused.',GEO='Existing free sole-edge geometryalpha0.2, reused.',THRESHOLD='Existing free residual cost relaxationalpha0.2, reused.',
            SUPPORT='Same new spatial correspondence and minimum support; always accept without image/FlyVis values.',RANDOM='Same support; hash(video,frame,tid,column,42) mapped uniformly to[-1,1], independent of image.',
            IMAGE='Direct normalized receptor brightness and image derivatives.',R1='Direct causal temporal receptor features.',DIS='Existing ordinary motion on the exact same domain.',
            FLYVIS='Same fixed standardized8T4/T5 spatial maps.',PERM='Mean similarity under32fixed within-reference spatial permutations of those maps.'),
        evaluation='All641primary-known historical26queries and727shadow queries. Same original states and full native lifecycle for each one-step intervention. Reuse all existing complete updates if the entire first-match set is identical.',
        full_gate='Only after zero parity, source checks and independent conditional audit: selected system orSUPPORT has lower95%paired-video deltaJ vsB0>0 AND no increase in known wrong query assignments. No full rollout if this gate fails.',
        full_acceptance=dict(cohort='Separately registered same20historical val videos, detector-unseen but researcher-seen; no retuning.',
            bootstrap_samples=20000,seed=42,family=11,confidence_percent=100*(1-.05/11),
            primary_comparisons='Selected system-B0, SUPPORT-B0, FlyVis-eachother9variants; duplicated comparisons count conservatively in the fixed family11.',
            accept='Corrected lower pooled-HOTA bound>0, nonnegative mean-videoAssA/IDF1, no increase in totalIDSw/FP/FN or known wrong-target returns.',
            contribution='A useful selected system may be accepted againstB0 with its source stated. IncrementalFlyVis requires success against every matched control; noFlyVis claim from geometry or image.'),
        stop='If the preregistered gate fails, preserve result and do not deploy or mark problem1solved. Problems2–4remain deferred.',
        literature=[dict(url='https://arxiv.org/pdf/1703.07402',role='DeepSORT§2.2equations3–6: separate appearance admissibility and motion admissibility. It does not validate FlyVis as an identity descriptor.'),
            dict(url='https://arxiv.org/html/2110.06864',role='ByteTrack uses tracklet similarity and staged unmatched association; this verifier is a separate hypothesis.')])
    write(RUN/'protocol.json',protocol,new=True)
    path=ROOT/'experiments/project_history.json';catalog=read(path);old=next(c for c in catalog['experiments'] if c['id']=='singleton_free_cost_v1_20261108')
    assert old['state']=='running' and not old['obtained']
    old.update(state='completed_negative',done='2026-11-08',
        obtained='Нулевой вариант точно совпал на1204кадрах;5тестов операции прошли (runs/singleton_free_20261108/technical.json). На641запросе: B0=41правильный/6чужих, свободная геометрия=51/11, FlyVis=51/10, изображение=51/10. Нижняя95%граница deltaJдляFlyVis=0; строгий критерий>0не пройден, полный пилот не запускался (runs/singleton_free_20261108/comparison_report.json). Аудит17919строк и14256матриц прошёл (independent_audit.json).',
        interpretation='Фиксация штатных пар сама по себе не решила проблему чужой единственной рамки. В14из15новых известных назначений относительный зрительный счёт не применялся из-за отсутствия второго трека-претендента (runs/singleton_free_20261108/free_verification_diagnosis.json).',
        decision='Версию не принимать. Следующая правка первой проблемы — отдельная проверка сходства с собственным прошлым шаблоном; остальные проблемы отложены.')
    for a,proof in zip(old['acceptance'],['technical.json','selection.json','comparison_report.json']):a.update(met=proof!='comparison_report.json',evidence='runs/singleton_free_20261108/'+proof)
    for name in ['comparison_report.json','independent_audit.json','technical.json','free_verification_diagnosis.json']:
        old['evidence'].append(dict(kind='file',ref='runs/singleton_free_20261108/'+name,claim='Проверенный отрицательный этап первой проблемы'))
    catalog['experiments'].append(dict(id=protocol['id'],title='Единственная свободная рамка — проверка по собственной прошлой карте цели',area='flyvis',state='planned',started='2026-11-08',
        question='Можно ли отклонять чужую единственную рамку сравнением с собственной прошлой целью без второго трека-претендента?',
        hypothesis='Абсолютное пространственное сходство может проверять личность там, где относительный счёт между треками не определён.',
        expected='Меньше чужих возвратов при сохранении правильных; контроли отделяют данные изображения, обычное движение, геометрию и FlyVis.',
        acceptance=[dict(criterion='Полный нулевой выход и все штатные первые пары сохранены.',met=None),dict(criterion='Порог и выбранная система заморожены только поfit20до26.',met=None),
            dict(criterion='Полный пилот только по зарегистрированному критерию; принятие поHOTAи ограничениям вреда.',met=None)],obtained='',interpretation='',decision='',
        cannot_claim='Тот же исторический материал. Геометрическая допустимость, отсутствие конкурента и пространственная корреляция не гарантируют личность. Принцип проверки внешности заимствован из прежних трекеров.',
        evidence=[dict(kind='file',ref='runs/singleton_verify_20261108/protocol.json',claim='Конкретное действие, первоисточники, контроли, настройка и остановка до новых карт'),
            dict(kind='file',ref='runs/singleton_verify_20261108/inventory_recovery1.json',claim='589последовательностей; готовые карты переиспользуются')]))
    write(path,catalog)
    entry=f"**{datetime.now(MSK).strftime('%d.%m.%Y, %H:%M MSK')}** — Ограниченная свободная правка `singleton_free_cost_v1_20261108` не прошла критерий перехода: на641запросе геометрия добавила10правильных/5чужих назначений, FlyVisи изображение10/4; нижняя95%границаFlyVis−B0равна0 (`runs/singleton_free_20261108/comparison_report.json`). Независимые17919меток/14256матриц/11интервалов проверены (`independent_audit.json`). Полный пилот не запускался. В14из15новых известных назначений зрительная поправка не определена; у всех этих случаев только один трек-претендент (`free_verification_diagnosis.json`).\n\n*Следующее действие первой проблемы.* Зарегистрирована проверка единственного кандидата по собственному прошлому шаблону `singleton_own_template_v1_20261108`, по принципу отдельной допустимости внешности DeepSORT§2.2(https://arxiv.org/pdf/1703.07402), без заявления о пригодностиFlyVisдо измерения. Геометрическая скидка0,2и штатные пары сохраняются; сигнал только разрешает или запрещает новую свободную пару. Общий контроль пространственной поддержки, случайное прореживание, изображение, прямой временной вход, обычное движение и перемешанные карты отделяют причины эффекта. Порог настраивается только наfit20, включая чужие занятые рамки как офлайн-примеры; runtimeих не отбирает. Найдены589необходимыхпоследовательностей:193готовыхполныхнабора,134точныхнынешнихкэша,175новыхшаблонов и87новыхтекущихпоследовательностей (`runs/singleton_verify_20261108/inventory_recovery1.json`). Прежний частичныйinventory.jsonс ошибкой сериализацииnumpy.boolсохранён; исправленный отчёт отдельный. Протокол, ограничения и остановка — `runs/singleton_verify_20261108/protocol.json`. Остальные три проблемы отложены.\n\n"
    write(RUN/'registration_journal.json',dict(entry=entry),new=True)
    plan=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';text=plan.read_text(encoding='utf-8');anchor='## Журнал изменений\n\n';assert text.count(anchor)==1
    plan.write_text(text.replace(anchor,anchor+entry,1),encoding='utf-8')
    sources={}
    for name in ['common.py','inventory.py','prepare_inputs.py','queue_data.py','catalog_check.py','register.py']:
        p=Path(__file__).parent/name;ast.parse(p.read_text(encoding='utf-8-sig'));sources[p.relative_to(ROOT).as_posix()]=sha(p)
    for name in ['tracking/foveated_eyes.py','tracking/retina.py','tracking/flyvis_adapter.py','runs/z13_local_peak_20261021/geometry.json',
        'runs/identity_assignment_20261108/export_complete.json','runs/identity_assignment_20261108/input_complete.json',
        'runs/singleton_verify_20261108/inventory_recovery1.json','runs/singleton_verify_20261108/protocol.json']:
        sources[name]=sha(ROOT/name)
    write(RUN/'input_phase.json',dict(registered=datetime.now(MSK).isoformat(),sources=sources),new=True)
    status('registered_first_problem_own_template',deadline=protocol['deadline'])
    with (RUN/'queue.log').open('x',encoding='utf-8') as out,(RUN/'queue.err').open('x',encoding='utf-8') as err:
        proc=subprocess.Popen([sys.executable,'-u','research_singleton_verify_20261108/queue_data.py'],cwd=ROOT,stdout=out,stderr=err,
            creationflags=subprocess.CREATE_NO_WINDOW|subprocess.IDLE_PRIORITY_CLASS)
    root=dict(pid=proc.pid,created=psutil.Process(proc.pid).create_time(),script='research_singleton_verify_20261108/queue_data.py')
    write(RUN/'queue_started.json',root,new=True);roots=read(BASE/'deadline_roots.json');roots['roots'].append(root)
    for attempt in range(5):
        try:write(BASE/'deadline_roots.json',roots);break
        except PermissionError:
            if attempt==4:raise
            time.sleep(.1)
    write(RUN/'deadline_registered.json',dict(passed=True,root=root,deadline=protocol['deadline']),new=True)


if __name__=='__main__':main()
