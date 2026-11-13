"""Inspect all retained held-out masks against fixed, outcome-blind cores."""
from pathlib import Path
import json,hashlib,csv
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE.parent
read=lambda p:json.loads(p.read_text(encoding='utf8'))
manifest=read(OUT/'holdout/manifest.json')
result=read(OUT/'stage_heldout/results.json')
bank=np.load(OUT/'stage_heldout/masks.npz')
rows=[];summary=[]
for event in manifest['events']:
    roles={role['role']:role for role in event['roles']}
    source=np.load(ROOT/event['masks'])
    for segment in [s for s in result['segments'] if s['event']==event['id']]:
        role=roles['target_ref'] if 'target_ref' in segment['roles'] else roles['query']
        target=source[role['manual_key']];other=source[role['other_manual_key']]
        current=[]
        for index,mask in enumerate(bank[segment['mask_bank_key']]):
            overlap=int(np.count_nonzero(mask&target));leak=int(np.count_nonzero(mask&other));total=int(mask.sum())
            row=dict(event=event['id'],frame=segment['frame'],mask_index=index,score=segment['scores'][index],bbox_xyxy=segment['boxes'][index],selected_roles=[r for r,chosen in zip(segment['roles'],segment['selected']) if chosen==index],mask_pixels=total,target_core_pixels=int(target.sum()),target_intersection=overlap,target_core_coverage=overlap/int(target.sum()),other_core_pixels=int(other.sum()),other_intersection=leak,other_core_coverage=leak/int(other.sum()),mask_target_fraction=overlap/total if total else None,mask_other_fraction=leak/total if total else None)
            rows.append(row);current.append(row)
        summary.append(dict(event=event['id'],frame=segment['frame'],any_retained_mask_contains_target_core=any(row['target_intersection']>0 for row in current),mask_indices_with_target_core=[row['mask_index'] for row in current if row['target_intersection']>0],mask_indices_with_target_but_no_observed_other_core=[row['mask_index'] for row in current if row['target_intersection']>0 and row['other_intersection']==0],all_retained_masks_enumerated=len(current)))
report=dict(passed=True,scope='Post-hoc diagnostics only; all masks retained under fixed conf0.25, no new inference/selection rule.',rows=rows,summary=summary,outputs_sha256=hashlib.sha256((OUT/'stage_heldout/masks.npz').read_bytes()).hexdigest(),caveat='Zero intersection with partial observed core does not describe unknown contour pixels. A missing retained detection does not prove absence before confidence filtering.')
(HERE/'heldout_all_auto_masks_oracle_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
with (HERE/'heldout_all_auto_masks_oracle_table.csv').open('w',newline='',encoding='utf8') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print(json.dumps(summary,ensure_ascii=False))
