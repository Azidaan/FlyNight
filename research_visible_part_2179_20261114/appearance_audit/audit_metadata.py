"""Read-only metadata/source audit; no model imports or inference."""
import hashlib
import json
import math
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
OLD = ROOT / 'research_assignment_weights_2179_20261112'
MEM = ROOT / 'research_memory_2179_20261113'

def read(path):
    return json.loads(path.read_text(encoding='utf8'))

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def location(path, needles):
    lines = path.read_text(encoding='utf8').splitlines()
    return [dict(line=i+1, text=line) for i, line in enumerate(lines)
            if any(needle in line for needle in needles)]

def level(box):
    scale = math.sqrt((box[2]-box[0])*(box[3]-box[1])) * .8
    return dict(resized_geometric_size=scale,
                level=max(0,min(3,math.floor(math.log2(scale/56 + 1e-6)))))

rt = read(OLD / 'runtime.json')
routes = []
for event in rt['events']:
    routes.append(dict(
        frame=event['frame'],
        references=[dict(tid=row['tid'],frame=row['reference']['frame'],
                         box=row['reference']['box'],**level(row['reference']['box']))
                    for row in event['rows'] if row['tid'] in [2179,2445]],
        candidates=[dict(raw_id=c['raw_id'],box=c['box'],**level(c['box']))
                    for c in event['candidates'] if c['raw_id'] in [109555,109571,109699,110260]]))

dominance = read(OLD / 'pairwise_dominance.json')['records']
delta = 0.00010757893323898315
component_bound = math.sqrt(256)*delta
bounds=[]
for record in dominance:
    costs=record['endpoints']['appearance_1.0']
    margin=costs['2179']-costs['2445']
    bounds.append(dict(frame=record['frame'],measured_appearance_margin=margin,
                       reference_l2_bound=component_bound,
                       two_edge_margin_bound_fixed_unit_query=component_bound,
                       margin_exceeds_bound=margin>component_bound))

evaluations=read(OLD / 'result.json')['evaluation_tables']
anchor=next(e for e in evaluations if e['frame']==1159)
assert anchor['current']['109555']==['00122746','00122729']
trace=read(MEM / 'trace_v2.json')
row=next(r for r in trace['rows'] if r['frame']==1159 and r['tid']==2445)

source_paths=[ROOT/'tracking/masa_r50.py',ROOT/'tools/masa_ref/deform_fusion.py',
              ROOT/'tools/masa_ref/masa_track_head.py',ROOT/'tools/masa_ref/masa.py',
              OLD/'runtime.json',OLD/'result.json',OLD/'pairwise_dominance.json',
              MEM/'feature_recovery_protocol.json',MEM/'feature_result.json',
              MEM/'extract_v2.py',MEM/'trace_v2.json']
source_hashes={p.relative_to(ROOT).as_posix():digest(p) for p in source_paths}
source_lines={
    'tracking/masa_r50.py':location(ROOT/'tracking/masa_r50.py',[
        'self.norm = nn.GroupNorm','def features(','def embed_boxes(',
        'area = ','lvl = ','roi_align(', 'def preprocess(', '[:, :, ::-1]',
        'offset.contiguous().reshape', 'mask.contiguous().reshape']),
    'tools/masa_ref/deform_fusion.py':location(ROOT/'tools/masa_ref/deform_fusion.py',[
        'type="GN"','if not self.fix_upsample','spatial_conv_high(x[level + 1]'])}
result=dict(
    passed=True,checked_utc=datetime.now(timezone.utc).isoformat(),
    scope='Read-only source and saved-result audit; no model, CUDA or full-tracker execution.',
    source_hashes=source_hashes,source_lines=source_lines,routing=routes,
    exact_anchor_ambiguity=dict(frame=1159,tid=2445,raw_id=109555,
        expected_from_past=anchor['expected']['2445'],
        current_unique_hit_rule=anchor['current']['109555'],
        exact_overlap_values=row['detection_ownership']),
    batch_error_bounds=dict(component_max_error=delta,dimension=256,records=bounds,
        limitation='Analytical bound for reported repeated reference vectors and fixed unit query. Not an empirical bound on every unseen ROI. Misranking predates failed memory extraction; final matched-batch overlap error is0.'),
    verified_official_config_agreement=['RGB conversion','ImageNet mean/std',
        'long side1024','padding divisor32','ResNet50 caffe stride placement',
        'three DeformFusion blocks','four embedding convs','GN32 embedding normalization',
        'ROIAlign7x7 adaptive sampling','FPN size threshold56'],
    official_sources=[
        'https://raw.githubusercontent.com/siyuanliii/masa/main/configs/masa-one/masa_r50_plug_and_play.py',
        'https://raw.githubusercontent.com/open-mmlab/mmdetection/main/mmdet/models/roi_heads/roi_extractors/single_level_roi_extractor.py',
        'https://raw.githubusercontent.com/open-mmlab/mmcv/main/mmcv/ops/csrc/common/cuda/modulated_deform_conv_cuda_kernel.cuh'],
    limitations=['No full numerical parity test against official MMCV/MASA was run.',
        'ROI level mismatch is a confirmed routing difference, not yet a proven cause of wrong similarity.',
        'Whole-frame dependency is confirmed architecture; its measured contribution requires root pixel intervention.',
        'Current annotation overlap cannot identify visible pixels or prove unique membership.'])
target=OUT/'audit.json'
with target.open('x',encoding='utf8',newline='\n') as f:
    json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
print('APPEARANCE_METADATA_AUDIT_COMPLETE',len(routes),len(source_hashes))
