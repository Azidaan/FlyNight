"""Новый короткий режим до просмотраval; не возобновляет матрицу истории."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261028 import RUN,service


def main():
    folder=RUN/'b2_sparse_training_pilot';checks=RUN/'b2_sparse_training_preparation'
    assert not folder.exists()
    c=service.read(RUN/'b2_causal960_record/controller_result.json')
    assert c['exit_code']==0 and c['all_owned_ended'] and not any(service.matching(k,v) for k,v in c['owned'].items())
    assert service.read(RUN/'b2_causal960_record/report.json')['passed']
    data=service.read(RUN/'b2_causal_manifest/report.json')
    entries=[]
    for e in data['entries']:
        assert service.sha(ROOT/e['manifest'])==e['sha256']
        m=service.read(ROOT/e['manifest']);n=len(m['clips'])
        # Integer evenly spaced positions, including both boundaries; no GT or
        # observed quality enters selection. Exactly10positions at each video.
        assert n>=10
        indices=[(i*(n-1))//9 for i in range(10)]
        assert len(set(indices))==10
        inputs=[m['video_path'],m['alignment'],m['gt_only_detection_targets'],
                f"runs/train_cache/{e['video']}/detections/raw_video.npz",f"runs/train_cache/{e['video']}/detections/raw_video.json"]
        entries.append(dict(video=e['video'],split=e['split'],manifest=e['manifest'],manifest_sha256=e['sha256'],
                            clip_indices=indices,input_sha256={f:service.sha(ROOT/f) for f in inputs}))
    files=['scripts/train_b2_sparse_pilot_20261029.py','tracking/b2_causal_detector.py','tracking/b2_feature_temporal.py',
           'tracking/b1_learning.py','tracking/z13_readout.py','tracking/b1_fast_stimulus.py','tracking/b1_index_copy_stimulus.py',
           'scripts/z13_target_model.py','weights/baseline_seed42/best.pt','runs/flyvis/results/flow/0000/000/best_chkpt',
           'runs/continuation_20261028/b2_causal960_profile/report.json','runs/continuation_20261028/b2_causal960_record/report.json']
    protocol=dict(id='b2_sparse_training_pilot_v1',created=service.now(),budget_seconds=3600,seed=26,epochs=1,
        arms=['C','D','R1','B','last','mean'],entries=entries,train_clips=840,dev_clips=200,
        sampling='Exactly10uniformtargetqueryindices per each104historicalfit84/fit20video,includingfirst/lastavailable5queryclip; '
                 'chosen by compute budget BEFORE any val or quality result. Not a complete20441clip training run.',
        action='Same fused frozen YOLO26s960,common4609parameter input/output adapters,additive .1P3correction before one-to-one head. '
               'Separate stock core C/D,only D trains734coreparameters; R1 fixed; B/last/mean same independently trained9602parameterHexGRUinitialization.',
        initialization='Adapters identical withinseed26,outputzero; C/Dstockcoreequalbutseparateparameters; B/last/meaninitialweightsidentical. '
                       'No promotion of older3scratchsteps or diagnosticweights. One epoch,Adamadapter/B lr.001,coreoriginalCORE_LR,clipgrad1,corepenaltyandclampasoldrealpath.',
        causality='GT only lastquerytargets,identicalfixedsample/history/allcontrols,noGTinput. Real fivequerytimestamps/FPS, '
                  'FlyVisdt<=.02,TimedR1decayatrealintervalcalibratedtoregular.2s. No object/camera alignment claimed. '
                  'Frozenbackboneonceforeachneededquery,releasedP3afterlastselecteduse,no80GBfeatureexport.',
        loss='Native one-to-one v8DetectionLoss tal_topk1,LetterBoxintegerpadding/clipping; crowd anchors ignore only same-class background,positive target overridesignore; '
             'GT metadata/crowd never choose input/sampling. Dev detection loss is not detector accuracy.',
        price='After first COMPLETE fitvideo(all6models,10updates each) and COMPLETE firstdevvideo(costprobe),save checkpoint. '
              '2*(fit_seconds_per_clip*remaining_fit_clips+dev_seconds_per_clip*all_final_dev_clips)+180<remainingstage;false saves checkpoint/price,not scientificnegative. '
              'Finaldev reevaluatesfirstdevafteralltrainingbecauseweightschanged,notreuseofstale devloss.',
        scientific_acceptance_deferred='Training completion only. Before science freeze checkpoints and fullval detector/tracker protocol; '
             'D lowerpaired95%HOTA>0 to native,last,mean,C,R1,B is required for this one-seed pilot; no claim about fullB2/corewithoutmatchedcontrols. '
             'No historicvalmodel/epoch/alpha/thresholdtuning;oneepoch/seed/sparsepopulationfixedbeforeval.',
        priority='If a NEW direct human author_override for the stopped history matrix appears,checkpoint/end B2atvideo boundary;neveroverride090price implicitly.',
        sha256={f:service.sha(ROOT/f) for f in files})
    service.write(folder/'protocol.json',protocol)
    record=RUN/'b2_sparse_training_record';script='scripts/record_b2_sparse_pilot_20261029.py'
    service.write(record/'protocol.json',dict(budget_seconds=600,sha256={script:service.sha(ROOT/script)},created=service.now()))
    h=service.read(ROOT/'experiments/project_history.json');base=next(x for x in h['experiments'] if x['id']=='b2_causal960_path_v1')
    assert not any(x['id']==protocol['id'] for x in h['experiments'])
    h['experiments'].append(dict(id=protocol['id'],title='Б2 — короткое обучение временной ветви и контролей на104видео',
        area=base['area'],state='ready',started='2026-10-29',
        question='Можно ли обучить сопоставимые6ветвей Б2 в заранее ограниченном режиме до полногоval?',
        hypothesis='Настоящая ошибка детектора обучает общие адаптеры и734параметраDнапричинныхпятикадровыхвходах.',
        expected='840обновленийнакаждуюветвь,200devцелей,YOLO/Cнеизменны;качествотрекерапоканеоценено.',
        acceptance=[dict(criterion=s,met=None) for s in ['Одна полная ограниченная эпоха всех6ветвей,840fit/200devкаждой',
                    'Источники/одинаковаяинициализация/причинность/crowd/замороженныйYOLOиCпроверены',
                    'Конечныепотери/производные,ненулеваяградиент734параметровDпослепрогрева,контрольныеточкисхешами']],
        evidence=[dict(kind='file',ref=folder.relative_to(ROOT).as_posix()+'/protocol.json',claim='Режим/выборка/цена/ограничения дообучения')],
        cannot_claim='Это не обучение на всех20441клипах и не научная проверка HOTA;нетпользыядрапобезконтрольнойloss.'))
    service.write(ROOT/'experiments/project_history.json',h)
    from scripts.b123_safe_journal_20261027 import journal
    journal(protocol['id'],'Дообучения зарегистрирован ограниченный режим Б2 без выбора поval',
        'Источник runs/continuation_20261028/b2_sparse_training_pilot/protocol.json: C/D/R1/B/last/mean,однаэпоха seed26, '
        'по10равномерныхцелевыхзапросовнавсех104train/devвидео,840fit/200dev,неполные20441клип. '
        'Режим выбран по стоимости до качества/val;до полного научного HOTA нужны отдельные аудит/полныйтрекер/цена. '
        'GPUбюджет3600с,первыйполныйfit+devвидеогейт,нетпродленияматрицы/старыхбюджетов. ДоCUDAкаталог/62CPUтеста.')
    for i,args in enumerate([['scripts/validate_project_history.py'],['-m','pytest','tests/test_project_history.py','tests/test_b2_causal_detector.py','-q']]):
        with (checks/f'check_{i}.log').open('w',encoding='utf-8') as out:
            code=subprocess.run([sys.executable,*args],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,timeout=420).returncode
        assert code==0
    service.write(checks/'report.json',dict(passed=True,updated=service.now()))
    script='scripts/train_b2_sparse_pilot_20261029.py'
    service.write(RUN/'ready_jobs/116_b2_sparse_training_pilot.json',dict(name='b2_sparse_training_pilot',
                  folder=folder.relative_to(ROOT).as_posix(),script=script,script_sha256=service.sha(ROOT/script),protocol_sha256=service.sha(folder/'protocol.json')))


if __name__=='__main__':main()
