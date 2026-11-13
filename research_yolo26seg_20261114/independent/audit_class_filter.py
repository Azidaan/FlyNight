"""Independent saved all-class diagnostic; original pipeline results untouched."""
from pathlib import Path
import json,hashlib,csv
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE.parent
read=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
data=read(OUT/'class_filter_result.json')
registration=read(OUT/'class_filter_protocol.json')
base_settings=read(OUT/'protocol.json')['settings']
expected=dict(base_settings,classes=None)
assert data['settings']==expected and base_settings['classes']==[2]
assert data['source_sha256']==registration['source_sha256']==sha(OUT/'class_filter_probe.py')
assert data['masks_sha256']==sha(OUT/'class_filter_masks.npz')
assert [row['image'] for row in data['records']]==registration['images']
bank=np.load(OUT/'class_filter_masks.npz')
orig=np.load(ROOT/'research_visible_part_2179_20261114/mask_review/masks_reviewed.npz')
new_manifest=read(OUT/'holdout/manifest.json')
rain=next(e for e in new_manifest['events'] if e['id']=='B_night_rain')
rain_masks=np.load(ROOT/rain['masks'])
base_results={}
base_banks={}
for stage in ['2179','heldout']:
    for row in read(OUT/('stage_'+stage)/'results.json')['segments']:base_results[row['image']]=row
    base_banks[stage]=np.load(OUT/('stage_'+stage)/'masks.npz')
rows=[];summary=[];parity=[]
for record in data['records']:
    path=record['image'];current=[]
    if 'research_visible_part' in path:
        frame=int(Path(path).stem);target=orig[f'{frame}_target'];other=orig[f'{frame}_neighbor'];occluder=orig[f'{frame}_occluder'];stage='2179'
    else:
        role=next(r for r in rain['roles'] if r['image']==path and r['role']!='competitor_ref')
        frame=role['frame'];target=rain_masks[role['manual_key']];other=rain_masks[role['other_manual_key']];occluder=None;stage='heldout'
    assert len(record['boxes'])==len(record['classes'])==len(record['scores'])==len(bank[record['key']])
    for index,mask in enumerate(bank[record['key']]):
        assert mask.dtype==np.bool_ and mask.shape==target.shape==(720,1280)
        cls=int(record['classes'][index]);area=int(mask.sum());t=int(np.count_nonzero(mask&target));o=int(np.count_nonzero(mask&other))
        row=dict(image=path,frame=frame,index=index,class_id=cls,class_name=data['names'][str(cls)],score=record['scores'][index],bbox_xyxy=record['boxes'][index],mask_pixels=area,target_core_intersection=t,target_core_coverage=t/int(target.sum()),other_core_intersection=o,other_core_coverage=o/int(other.sum()),occluder_intersection=int(np.count_nonzero(mask&occluder)) if occluder is not None else None,target_without_observed_other=t>0 and o==0,both_observed_cores=t>0 and o>0)
        assert row['score']>expected['conf']
        rows.append(row);current.append(row)
    original=base_results[path]
    car_indices=[i for i,c in enumerate(record['classes']) if int(c)==2]
    assert len(car_indices)==len(original['boxes'])
    for old_index,new_index in enumerate(car_indices):
        assert record['boxes'][new_index]==original['boxes'][old_index]
        assert record['scores'][new_index]==original['scores'][old_index]
        previous_mask=base_banks[stage][original['mask_bank_key']][old_index]
        different=int(np.count_nonzero(previous_mask!=bank[record['key']][new_index]))
        assert different==0,(frame,old_index,new_index,different)
        parity.append(dict(frame=frame,old_car_index=old_index,all_class_index=new_index,mask_pixels_different=different))
    usable=[r for r in current if r['target_without_observed_other']]
    summary.append(dict(image=path,frame=frame,all_class_masks=len(current),target_intersecting_masks=[dict(index=r['index'],class_name=r['class_name'],target_core_coverage=r['target_core_coverage'],other_core_coverage=r['other_core_coverage']) for r in current if r['target_core_intersection']>0],target_without_observed_other_masks=[dict(index=r['index'],class_name=r['class_name'],target_core_coverage=r['target_core_coverage']) for r in usable],class_filter_removal_restored_observed_target_separation=bool(usable) and not any(r['class_id']==2 for r in usable)))
report=dict(passed=True,source_sha256=sha(OUT/'class_filter_result.json'),mask_sha256=sha(OUT/'class_filter_masks.npz'),settings_change_only_classes=True,all_car_masks_exactly_reproduce=True,original_pipeline_results_unchanged=True,scope=data['scope'],all_masks=rows,per_frame=summary,car_output_parity=parity,caveats=['After class filter removal the confidence cutoff remains0.25; this cannot distinguish low confidence from absence of a target proposal.','Partial-core cleanliness is not a full contour accuracy estimate.','All-class exploratory observations cannot replace registered car-only held-out scores.'])
(HERE/'class_filter_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
with (HERE/'class_filter_all_masks_table.csv').open('w',newline='',encoding='utf8') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print(json.dumps(summary,ensure_ascii=False))
