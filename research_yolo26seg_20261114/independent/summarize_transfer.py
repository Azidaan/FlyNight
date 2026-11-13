from pathlib import Path
from datetime import datetime
import hashlib,json
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE.parent
read=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
audit=read(HERE/'heldout_audit_v2.json')
manifest=read(OUT/'holdout/manifest.json')
execution=read(OUT/'stage_heldout/execution_protocol.json')
assert datetime.fromisoformat(manifest['frozen_at'])<datetime.fromisoformat(execution['started'])
assert sha(OUT/'holdout/manual_masks.npz')==manifest['masks_sha256']
assert sha(OUT/'holdout/manual_polygons_frozen.json')==manifest['manual_polygons_sha256']
for name,h in execution['inputs'].items():assert sha(ROOT/name)==h
for event in manifest['events']:
    for role in event['roles']:assert sha(ROOT/role['image'])==role['image_sha256']
groups=[]
for arm in ['full','rectangle','manual','automatic']:
    for fill in ['imagenet_mean','black']:
        for level in [-1,0]:
            rows=[r for r in audit['comparisons'] if (r['arm'],r['fill'],r['level'])==(arm,fill,level)]
            groups.append(dict(arm=arm,fill=fill,level=level,event_count=len(rows),valid_target_preferred=sum(r['target_preferred'] for r in rows),raw_positive_margin_count=sum(r['margin_target_minus_competitor']>0 for r in rows),unavailable_count=sum(not r['all_required_masks_present'] for r in rows),events=[dict(event=r['event'],all_required_masks_present=r['all_required_masks_present'],target_preferred=r['target_preferred'],cosine_advantage=r['margin_target_minus_competitor'],derived_matching_cost_margin=r['margin_target_minus_competitor']/2,previous_gap_target_minus_competitor=-r['margin_target_minus_competitor']/2) for r in rows]))
out=dict(passed=True,scope='Three outcome-blind new diagnostic pairs on reserved video frames. These are not repairs of all original source old_id events and not a full tracker evaluation.',source_audit='heldout_audit_v2.json',source_audit_sha256=sha(HERE/'heldout_audit_v2.json'),manual_core_freeze_before_worker_seconds=(datetime.fromisoformat(execution['started'])-datetime.fromisoformat(manifest['frozen_at'])).total_seconds(),manual_inputs_and_images_unchanged=True,grouped_quality=groups,automatic_pipeline_times=[r for r in audit['timing'] if r['arm']=='automatic'],quality_gate_passed=False,latency_gate_all_cases_passed=False,reason='Full/rectangle/manual retain 3 valid correct pair preferences in each setting. Automatic retains 2; B_night_rain target reference has no assigned mask and current query mask contains competitor core but zero target-core pixels. Mean-fill positive margins for missing reference are rejected, not counted as a success. Automatic p95 exceeds registered200ms budget on day and rain.',scope_correction='All retained-output absence/merge claims so far concern classes=[2] (car), conf>0.25, max_det300. They do not yet distinguish class filtering or score filtering from segmentation model output; separate all-class probe must not replace this fixed-protocol result.',formulas=dict(cost='d=(1-cosine)/2',derived_matching_cost_margin='d_competitor-d_target=(cos_target-cos_competitor)/2',previous_gap='d_target-d_competitor=-derived_matching_cost_margin'),full_tracker_run=False)
(HERE/'transfer_summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(passed=True,manual_frozen_before_worker=out['manual_core_freeze_before_worker_seconds'],quality_gate_passed=False,latency_gate_all_cases_passed=False)))
