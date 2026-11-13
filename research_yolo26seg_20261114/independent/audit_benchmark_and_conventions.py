from pathlib import Path
import json,hashlib
from scalar_metrics import timing
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE.parent
def read(path):return json.loads(path.read_text(encoding='utf8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(name,obj):
    with (HERE/name).open('x',encoding='utf8') as stream:json.dump(obj,stream,ensure_ascii=False,indent=2,allow_nan=False)

audit=read(HERE/'2179_audit_v2.json')
rows=[]
for row in audit['comparisons']:
    item=dict(row)
    own,other=row['target_cosine'],row['competitor_cosine']
    item.update(cost_target=(1-own)/2,cost_competitor=(1-other)/2,derived_matching_cost_margin=(own-other)/2,previous_gap_target_minus_competitor=(other-own)/2)
    assert abs(item['derived_matching_cost_margin']-(item['cost_competitor']-item['cost_target']))<1e-12
    rows.append(item)
save('2179_margin_conventions.json',dict(passed=True,source='2179_audit_v2.json',source_sha256=sha(HERE/'2179_audit_v2.json'),formulas=dict(cosine='dot(a,b)/(norm(a)*norm(b))',cosine_advantage='cosine(query,target)-cosine(query,competitor); positive prefers target',matching_cost='d(a,b)=(1-cosine(a,b))/2',derived_matching_cost_margin='d(query,competitor)-d(query,target)=cosine_advantage/2; positive prefers target',previous_level_audit_gap='d(query,target)-d(query,competitor)=-cosine_advantage/2; negative prefers target'),interpretation='These are algebraic conventions for identical vectors, not a new experiment or a twofold improvement.',comparisons=rows))

data=read(OUT/'seg_only_benchmark.json')
addition=read(OUT/'benchmark_addendum.json')
assert data['passed'] and len(data['records'])==90
assert data['source_sha256']==addition['source_sha256']==sha(OUT/'benchmark_seg.py')
assert data['weight_sha256']==sha(ROOT/read(OUT/'weights_receipt.json')['path'])
assert addition['settings']==read(OUT/'protocol.json')['settings']
segments=read(OUT/'stage_2179/results.json')['segments']
mask_counts={frame:len(segment['boxes']) for frame,segment in zip([1159,1164,1165],segments)}
import numpy as np
mask_bank=np.load(OUT/'stage_2179/masks.npz')
mask_pixels={segment['frame']:int(mask_bank[segment['mask_bank_key']].sum()) for segment in segments}
summaries=[]
for index,frame in enumerate(addition['frames']):
    selected=[row for row in data['records'] if row['image_index']==index]
    assert len(selected)==30 and sorted(row['repeat'] for row in selected)==list(range(30))
    assert all(row['mask_count']==mask_counts[frame] and row['mask_pixels']==mask_pixels[frame] for row in selected)
    assert all(0<=row['peak_gpu_allocated_bytes']<=row['peak_gpu_reserved_bytes'] for row in selected)
    stats=timing([row['seconds']*1000 for row in selected])
    stats.update(frame=frame,peak_gpu_allocated_bytes=max(row['peak_gpu_allocated_bytes'] for row in selected),peak_gpu_reserved_bytes=max(row['peak_gpu_reserved_bytes'] for row in selected),peak_process_rss_bytes=max(row['process_rss_bytes'] for row in selected),mask_count=mask_counts[frame],mask_pixels=mask_pixels[frame])
    summaries.append(stats)
pooled=timing([row['seconds']*1000 for row in data['records']])
pooled.update(peak_gpu_allocated_bytes=max(row['peak_gpu_allocated_bytes'] for row in data['records']),peak_gpu_reserved_bytes=max(row['peak_gpu_reserved_bytes'] for row in data['records']),peak_process_rss_bytes=max(row['process_rss_bytes'] for row in data['records']))
receipt=dict(passed=True,source='research_yolo26seg_20261114/seg_only_benchmark.json',source_sha256=sha(OUT/'seg_only_benchmark.json'),per_frame=summaries,pooled=pooled,device=data['device'],scope=data['scope'],mask_totals_match_stage2179=True,limitations=['Pooled timing describes these three fixed frames, not unseen-video throughput.','PyTorch allocated/reserved peaks exclude driver/context allocations and other GPU applications.','Source inspection confirms CUDA synchronization and saved 10 warm-up/30 repeat configuration; independently recomputed from raw timings, no re-benchmark by auditor.'])
save('seg_only_benchmark_audit.json',receipt)
print(json.dumps(receipt,ensure_ascii=False))
