from pathlib import Path
import json,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent
old=np.load(HERE/'masks_core.npz',allow_pickle=False);arrays={}
mapping={'target':'red','neighbor':'black','occluder':'foreground','background':'background','uncertain':'uncertain'}
for frame in [1159,1164,1165]:
 for dst,src in mapping.items():arrays[f'{frame}_{dst}']=old[f'{frame}_{src}']
 stack=np.stack([arrays[f'{frame}_{k}'] for k in mapping]);assert stack.shape==(5,720,1280) and (stack.sum(axis=0)==1).all()
with (HERE/'masks_reviewed.npz').open('xb') as f:np.savez_compressed(f,**arrays)
receipt=json.loads((HERE/'masks.json').read_text(encoding='utf8'));receipt.update(file='masks_reviewed.npz',keys='{frame}_target, {frame}_neighbor, {frame}_occluder, {frame}_background, {frame}_uncertain',sha256=hashlib.sha256((HERE/'masks_reviewed.npz').read_bytes()).hexdigest(),review_status='Original images and diagnostic masks visually checked before any MASA outcomes. Five components form a disjoint exhaustive partition. Target/neighbor are conservative manually delimited visible cores, not complete contours.',reviewed_images=['1159_core_review.png','1164_core_review.png','1165_core_review.png'])
(HERE/'masks_reviewed.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf8')
print(receipt['sha256'])
