"""Known-answer checks for the independent measurement implementation."""
from pathlib import Path
import json
import numpy as np
from scalar_metrics import pair_iou,assignment_optimum,coverage,cosine,timing

assert pair_iou([0,0,2,2],[1,0,3,2])==1/3
assert pair_iou([0,0,2,2],[2,0,3,2])==0
# Two requested ROIs cannot receive the same single automatic instance.
choice=assignment_optimum([[0,0,2,2],[0,0,1,1]],[[0,0,2,2]])
assert choice['objective']==1 and choice['optimal_assignments']==[(0,-1)]
mask=np.asarray([[1,1,0],[0,1,0]],dtype=bool)
target=np.asarray([[1,0,0],[0,1,0]],dtype=bool)
other=np.asarray([[0,1,0],[0,0,1]],dtype=bool)
counts=coverage(mask,target,dict(neighbor=other))
assert counts['mask_pixels']==3 and counts['target_core_coverage']==1
assert counts['neighbor']['intersection']==1 and counts['neighbor']['core_coverage']==.5
assert cosine([1,0],[-1,0])==-1
assert cosine([1,0],[0,1])==0
assert cosine([0,0],[1,0]) is None
sample=timing([40,10,30,20])
assert sample['median_ms']==25 and sample['p95_ms']==38.5
receipt=dict(passed=True,checks=['continuous box IoU with touching-edge zero overlap','single-instance exclusivity independently optimized','known pixel core coverage and contamination','opposite/orthogonal/zero descriptor cosine','linearly interpolated sample median and p95'],inference=False)
(Path(__file__).parent/'scalar_known_answers.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
print(json.dumps(receipt))
