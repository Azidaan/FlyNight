"""Fixed numerical/assignment audit on saved matrices; no new map or rollout."""
from common import *
start('cpu',timeout=1200)
import gzip,collections
import numpy as np
from ultralytics.trackers.utils import matching
import lap
p=read(OUT/'protocol.json');phase=read(OUT/'boundary_protocol.json');verify(phase)
assert read(OUT/'comparison_report.json')['passed'] and read(OUT/'id_fate.json')['passed']
fates={(r['video'],r['arm'],r['frame'],r['old_id'],r['raw_id']):r for r in read(OUT/'id_fate.json')['records']}
def solve(C):return sorted(map(tuple,np.asarray(matching.linear_assignment(C,thresh=.8)[0]).reshape(-1,2).tolist()))
f32=np.array([[1.]],np.float32);f32[0,0]-=.2
unit=[]
for name,value in [('float32_subtraction',float(f32[0,0])),('float64_subtraction',1.-.2),
    ('one_float64_step_below',float(np.nextafter(np.float64(.8),-np.inf)))]:
    matrix=np.array([[value]],np.float64)
    unit.append(dict(name=name,value=value,greater_than_limit=value>.8,matches=solve(matrix)))
assert float(f32[0,0])>.8 and 1.-.2==.8
records=[];counts={};reproduced=0;maxdifference=0.
for arm in p['arms']:
    counter=collections.Counter();counts[arm]=counter
    for video in p['videos']:
        path=OUT/'rollouts'/arm/(video+'.events.json.gz');receipt=read(OUT/'rollouts'/arm/(video+'.receipt.json'))
        assert sha(path)==receipt['artifacts'][path.relative_to(ROOT).as_posix()]
        with gzip.open(path,'rt',encoding='utf8') as f:events=json.load(f)['events']
        for event in events:
            if not event['votes']:continue
            old=np.asarray(event['C'],np.float64);actual=np.asarray(event['corrected'],np.float64)
            actual_matches=solve(actual);assert actual_matches==sorted(map(tuple,event['actual_first']))
            reproduced+=1;promoted=old.copy()
            for i,j in event['votes']:promoted[i,j]-=.2
            difference=float(abs(promoted-actual).max());maxdifference=max(maxdifference,difference)
            alternative_matches=solve(promoted)
            counter['frames_with_accepted_proposals']+=1
            for i,j in event['votes']:
                proposal=next(x for x in event['proposals'] if x['row']==i and x['column']==j)
                key=(video,arm,event['frame'],proposal['old_id'],proposal['raw_id']);fate=fates[key]
                accepted=(i,j) in actual_matches;alternative=(i,j) in alternative_matches
                rec=dict(video=video,arm=arm,frame=event['frame'],old_id=proposal['old_id'],raw_id=proposal['raw_id'],
                    kind=fate['kind'],original_cost=float(old[i,j]),stored_corrected_cost=float(actual[i,j]),
                    float64_same_alpha_cost=float(promoted[i,j]),above_limit=bool(actual[i,j]>.8),
                    actual_global_selected=accepted,float64_global_selected=alternative,
                    old_ID_emitted=fate['old_id_emitted_on_candidate'])
                records.append(rec);counter['accepted_proposals']+=1
                counter['above_limit']+=int(rec['above_limit']);counter['global_selected']+=int(accepted)
                counter['float64_global_selected']+=int(alternative)
                counter['actual_unselected_but_not_above_limit']+=int(not accepted and not rec['above_limit'])
                if alternative and not accepted:counter['float64_added_'+fate['kind']]+=1
                if accepted and not alternative:counter['float64_lost_'+fate['kind']]+=1
        print('BOUNDARY',arm,video,dict(counter),flush=True)
verify(phase)
write('boundary_audit.json',dict(passed=True,lap_import_version=lap.__version__,lap_import_file=lap.__file__,
    unit_cases=unit,counts=counts,records=records,actual_global_frames_reproduced=reproduced,
    max_stored_vs_promoted_cost_difference=maxdifference,zero_model_or_image_calls=True,zero_tracker_rollouts=True,
    protocol_sha256=sha(OUT/'boundary_protocol.json'),
    limits='Float64 replay is a one-frame numerical counterfactual, not a causal video rollout or proof of saved IDs. Costs/alpha/threshold fixed before audit; no epsilon/threshold search. Actual recorded output remains the original pilot. Direct labels use only ready ID-fate evidence, unknowns unfilled.'))
journal('Проверена численная граница между считывателем и общим назначением',
    'Завершена отдельно зарегистрированная проверка установленногоLAPнаодноэлементныхматрицах и готовых полныхматрицах первого назначения, `runs/full_template_native_20261109/boundary_audit.json`. Никакие карты, модели, видео или пороги не пересчитывались. Восстановление техже фактическихназначений проверено; арифметикаfloat64с прежнейпоправкой0,2рассматривается толькокакоднокадровыйконтрфактическийразбор, не каксохранённыеID. Числа ивыводыдляисправленияподлежат отдельномуразборупослеитоговпилота.')
print('BOUNDARY AUDIT COMPLETE',json.dumps(cast(counts)),flush=True)
