"""Post-hoc oracle diagnostic only: enumerate every saved automatic mask."""
from pathlib import Path
import csv,hashlib,json
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE.parent
read=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
result=read(OUT/'stage_2179/results.json')
bank=np.load(OUT/'stage_2179/masks.npz')
manual=np.load(ROOT/'research_visible_part_2179_20261114/mask_review/masks_reviewed.npz')
rows=[]
summaries=[]
for group in result['segments']:
    frame=group['frame']
    parts={name:manual[f'{frame}_{name}'] for name in ['target','neighbor','occluder','background','uncertain']}
    assert np.all(np.stack(list(parts.values())).sum(axis=0)==1)
    current=[]
    for index,mask in enumerate(bank[group['mask_bank_key']]):
        total=int(mask.sum())
        row=dict(frame=frame,index=index,score=group['scores'][index],bbox_xyxy=group['boxes'][index],selected_for_roles=[role for role,chosen in zip(group['roles'],group['selected']) if chosen==index],mask_pixels=total)
        for name,part in parts.items():
            overlap=int(np.count_nonzero(mask&part))
            row[name+'_pixels']=overlap
            row[name+'_core_coverage']=overlap/int(part.sum())
            row[name+'_fraction_of_mask']=overlap/total if total else None
        row['target_minus_neighbor_core_coverage']=row['target_core_coverage']-row['neighbor_core_coverage']
        row['observed_both_vehicle_cores']=row['target_pixels']>0 and row['neighbor_pixels']>0
        row['target_present_without_observed_neighbor_core']=row['target_pixels']>0 and row['neighbor_pixels']==0
        row['target_present_without_observed_neighbor_or_occluder']=row['target_pixels']>0 and row['neighbor_pixels']==row['occluder_pixels']==0
        assert sum(row[name+'_pixels'] for name in parts)==total
        rows.append(row);current.append(row)
    def best(items,key):
        if not items:return []
        maximum=max(row[key] for row in items)
        return [row['index'] for row in items if row[key]==maximum]
    clean=[row for row in current if row['target_present_without_observed_neighbor_core']]
    clean_both=[row for row in current if row['target_present_without_observed_neighbor_or_occluder']]
    summaries.append(dict(frame=frame,raw_mask_count=len(current),selected=group['selected'],roles=group['roles'],target_without_observed_neighbor_indices=[row['index'] for row in clean],target_without_observed_neighbor_or_occluder_indices=[row['index'] for row in clean_both],highest_target_core_coverage_indices=best(current,'target_core_coverage'),highest_target_minus_neighbor_coverage_indices=best(current,'target_minus_neighbor_core_coverage'),highest_target_core_coverage_without_neighbor_indices=best(clean,'target_core_coverage'),target_neighbor_separation_available_in_saved_outputs=bool(clean),caveat='Zero overlap with partial neighbor core does not prove a complete pure-object mask; uncertain and unannotated visible pixels remain.'))
report=dict(passed=True,scope='Post-hoc diagnostic using manual partial cores to inspect already saved masks, not a deployable selector or an independent model-selection test.',mask_bank_sha256=sha(OUT/'stage_2179/masks.npz'),manual_core_sha256=sha(ROOT/'research_visible_part_2179_20261114/mask_review/masks_reviewed.npz'),all_masks=rows,per_frame=summaries,selection_unchanged=True,new_inference=False,no_score_threshold_search=True,formula='Core coverage = intersection(mask,core)/core area; difference = target coverage - neighbor coverage. Zero-overlap predicates are literal observed pixel intersections, not fitted thresholds.',limitation='These observations localize missing separation to postprocessed retained model outputs; they do not rule out a discarded low-confidence detection or a different training regime.')
(HERE/'2179_all_auto_masks_oracle_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
with (HERE/'2179_all_auto_masks_oracle_table.csv').open('w',newline='',encoding='utf8') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print(json.dumps(dict(per_frame=summaries,rows=rows),ensure_ascii=False))
