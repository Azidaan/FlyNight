"""Заморозить научный трекерный пилот Б2 до первого valвыхода."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
from datetime import datetime
import sys,subprocess,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261028 import RUN,service


def main():
    folder=RUN/'b2_tracker_pilot';checks=RUN/'b2_tracker_preparation'
    for name in ('b2_sparse_training_pilot','b2_sparse_training_record','b2_sparse_training_audit'):
        c=service.read(RUN/name/'controller_result.json')
        assert c['exit_code']==0 and c['all_owned_ended'] and not any(service.matching(k,v) for k,v in c['owned'].items())
    audit=service.read(RUN/'b2_sparse_training_audit/report.json')
    if not audit['passed']:
        service.write(checks/'report.json',dict(ready=False,reason='Trainingauditnotcomplete;noGPUvalidation',scientific_verdict=None,updated=service.now()));return
    assert not folder.exists()
    report=service.read(RUN/'b2_sparse_training_pilot/report.json');assert report['complete']
    checkpoint=report['final'];assert service.sha(ROOT/checkpoint['path'])==checkpoint['sha256']
    old=service.read(RUN/'history_tracker_pilot/protocol.json');assert len(old['sources'])==40
    freeze=dict(checkpoint=checkpoint['path'],checkpoint_sha256=checkpoint['sha256'],training_report_sha256=service.sha(RUN/'b2_sparse_training_pilot/report.json'),
                independent_training_audit_sha256=service.sha(RUN/'b2_sparse_training_audit/report.json'),seed=26,epochs=1,frames=5,correction_scale=.1,created=service.now())
    service.write(folder/'frozen.json',freeze)
    files=['scripts/run_b2_full_tracker_20261029.py','scripts/audit_b2_full_tracker_20261029.py','scripts/summarize_b2_tracker_20261029.py',
           'tracking/b2_pilot_runtime.py','tracking/b2_causal_detector.py','tracking/b2_feature_temporal.py',
           'weights/baseline_seed42/best.pt',checkpoint['path'],(folder/'frozen.json').relative_to(ROOT).as_posix()]
    for s in old['sources']:files.extend(s[k] for k in ('video_path','detections','alignment','baseline','raw_array'))
    protocol=dict(id='b2_predetection_tracker_sparse_v1',created=service.now(),budget_seconds=13000,
        gpu_absolute_end='2026-10-29T04:48:00+03:00',audit_absolute_end='2026-10-29T07:18:00+03:00',
        sources=old['sources'],tracker_config=old['tracker_config'],arms=['native','zero','C','D','R1','B','last','mean'],
        inference='Same frozen YOLO26s960/.01/.7/FP32;realfivequerywindowsatallprioralignmentqueries,initialfour unchanged. '
                  'Commonprefixonceperquery;separatelearnedweights/states;true boxes/scores before full30Hztracker;otherframesoriginalcached detections. '
                  'All8trackers independentID/state;zero actually computesD then discards correction before detection head;native/zero raw/JSON/mean/cov exact every frame.',
        training='One epoch seed26,840train/200devtargets,10uniformtargetsper104historicaltrain/devvideo. '
                 'Sparse sampling chosenbycompute BEFOREval;noepoch/threshold/checkpoint/alpha selectiononval. NoclaimfullB2.',
        statistics='D HOTA pairedvideoBootstrap20k/seed1 to native,C,R1,B,last,mean;all6adjustedBonferronilowerbounds>0 primary;nominal95%also shown. '
                   'FullHOTA/DetA/AssA/IDF1/FP/FN/IDSw;systemD-native separatefromtrainedcoreD-C;noadditionofeffects.',
        raw_head_cpu_gpu_tolerance=.001,postprocess_box_tolerance_pixels=.002,postprocess_score_tolerance=1e-6,
        numerical_audit='BEFOREval:allCPUone-to-one rawbox/scoreelements comparedtoGPUFP32 within.001 absolute(logits/rawdist); '
            'fullCNNCPU/CUDA convolutions notassertedbitexact. Then savedGPUraw decodedindependently:anchors/classes/rankwithin1e-6,boxeswithin.002px. '
            'FinalGPUdetections checked,then usedforEXACTdiscreteCPUtracker/raw/JSON/state/HOTA/pairedreplay. ThisdoesnotchangeGPUquality/thresholds. '
            'Olderhistoryhead3e-5tolerance remainsunchanged and unrelated.',
        price='After first COMPLETE8variantvideoincludingHOTA/save:2*time*39+600<remaining13000and2*savedbytes*39+5GB<freeDisk. '
              'False meanssavefirst/remaining,norepeat/noextension/noscientificnegative. RawGPUheads/outputs/JSON savedBEFOREmetrics;recoverCPUmetricswithoutGPUrepeat. '
              'Audit9000CPUfirstCOMPLETEvideo2*time*39+300<remaining;checkpointsaftereachcompleteauditedvideo with allsourceSHA.',
        limitations='Historical40valnotnewindependentfinaltest,oneepoch/seed/sparsetrain;fixedP3notmotionaligned,fivequeryresetwindows,unequalcorecontrolparametercounts; '
                    'lossnotquality,CPUnumericaltoleranceexplicit. No newMASAbranch/data/weights/environment. Mainhypothesisgraph11texts/colorsprotected.',
        sha256={f:service.sha(ROOT/f) for f in sorted(set(files))})
    service.write(folder/'protocol.json',protocol)
    for name,budget in [('b2_tracker_audit',9000),('b2_tracker_record',600)]:
        service.write(RUN/name/'protocol.json',dict(budget_seconds=budget,parent=protocol['id'],created=service.now()))
    h=service.read(ROOT/'experiments/project_history.json');base=next(c for c in h['experiments'] if c['id']=='b2_sparse_training_pilot_v1')
    # Preserve prelaunch limitation text as required by AGENTS3.4a. The two
    # earlier infrastructure recorders had replaced this field without a copy;
    # recover the exact registered text, never invent a prior scientific result.
    old_limits={'b2_causal960_path_v1':service.read(RUN/'b2_causal960_profile/protocol.json')['limitations'],
                'b2_sparse_training_pilot_v1':'Это не обучение на всех20441клипах и не научная проверка HOTA;нетпользыядрапобезконтрольнойloss.'}
    for cid,old_text in old_limits.items():
        prior=next(c for c in h['experiments'] if c['id']==cid)
        if not any(x.get('field')=='cannot_claim' and x.get('old')==old_text for x in prior.get('revisions',[])):
            prior.setdefault('revisions',[]).append(dict(date='2026-10-29',commit=None,field='cannot_claim',old=old_text,
                reason='Восстановлен точный текст ограничения до инфраструктурного прогона; прежний recorder не сохранил копию поля. Исходный протокол/скрипт сохранены.'))
    assert not any(c['id']==protocol['id'] for c in h['experiments'])
    h['experiments'].append(dict(id=protocol['id'],title='Б2 — обученная временная ветвь детектора в полном трекере',area=base['area'],state='ready',started='2026-10-29',
        question='Даёт ли изменение признаков детектора пользу полному трекеру сверх сильных контролей?',
        hypothesis='Пять реальных queryкадров до головы детектора могут изменить обнаружения и полный трекер;вклад734обученныхпараметровизолируетсяD-C.',
        expected='Полныйсопоставимыйпилот40val/8вариантов,нулевойконтрольточен,полныйCPUаудит/метрики/парныеинтервалы.',
        acceptance=[dict(criterion=s,met=None) for s in ['Все40val/8вариантовполностьювыполненыбезGTвпредикторе',
            'ПолныйнезависимыйCPUаудитвходов/всехrawhead/назначений/выходов/метрик,взаранеезаданномчисленномдопуске',
            'Native/zero/raw/JSON/mean/covточныштатномуполномустенду',
            'Все6нижнихскорректированныхграницHOTA D кnative/C/R1/B/last/meanположительны']],
        evidence=[dict(kind='file',ref=folder.relative_to(ROOT).as_posix()+'/protocol.json',claim='Доvalзамороженывеса/выборка/пороги/статистика/численныйаудит/бюджет')],
        cannot_claim=protocol['limitations']))
    service.write(ROOT/'experiments/project_history.json',h)
    from scripts.b123_safe_journal_20261027 import journal
    journal(protocol['id'],'Доval заморожен полный трекерный пилот Б2',
            'runs/continuation_20261028/b2_tracker_pilot/protocol.json:40историческихval/native/zero/C/D/R1/B/last/mean, '
            'настоящиеqueryобнаруженияпередполным30Гцтрекером,6первичныхDсравнений/Bonferroni/95%ДИ. '
            'Новые13000GPU/9000CPU/600record,первыйполныйвидеопроход/диск/цена,никакогопродленияматрицыистории. '
            'CPUвсеrawголовполногоYOLOвдопуске.001доval,postprocess.002px/1e-6score;дискретныйповторпослепроверкииспользуетGPUдетекции. '
            'Предыдущее3e-5уисторииНЕменялось. 1эпоха/seed26/разреженныйfit—ограничение. '
            'Также восстановлены точные прежние cannot_claim двух инфраструктурных карточек в revisions по исходным протоколам/скриптам: recorder ранее не сохранил копии этих полей. ДоCUDAкаталог/64tests.')
    for i,args in enumerate([['scripts/validate_project_history.py'],['-m','pytest','tests/test_project_history.py','tests/test_b2_causal_detector.py','tests/test_b2_tracker_runtime.py','tests/test_b2_tracker_summary.py','-q']]):
        with (checks/f'check_{i}.log').open('w',encoding='utf-8') as out:
            code=subprocess.run([sys.executable,*args],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,timeout=420).returncode
        assert code==0
    save=datetime.fromisoformat(service.read(RUN/'authorization.json')['save_at']).timestamp()
    fits=save-time.time()>13000+9000+600+120
    service.write(checks/'report.json',dict(ready=fits,passed=True,full_chain_fits_before_save=fits,updated=service.now()))
    if not fits:return
    for index,name,script in [('130','b2_tracker_pilot','scripts/run_b2_full_tracker_20261029.py'),
                              ('135','b2_tracker_audit','scripts/audit_b2_full_tracker_20261029.py'),
                              ('138','b2_tracker_record','scripts/record_b2_full_tracker_20261029.py')]:
        target=RUN/name
        service.write(RUN/f'ready_jobs/{index}_{name}.json',dict(name=name,folder=target.relative_to(ROOT).as_posix(),script=script,
                      script_sha256=service.sha(ROOT/script),protocol_sha256=service.sha(target/'protocol.json')))


if __name__=='__main__':main()
