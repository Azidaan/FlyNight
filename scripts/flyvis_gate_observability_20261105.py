"""Невозможность различить движение и покой по однородным кадрам."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/flyvis_still_gate_20261105';sys.path.insert(0,str(ROOT))
from scripts.flyvis_offset_recovery2_20261104 import frames_for,positions
from scripts.flyvis_still_gate_20261105 import stationary
def main():
 gray=np.full((720,1280),127,np.uint8)
 still=dict(start='still',speed=1,axis=0,contrast=1.0);moving=dict(start='moving',speed=1,axis=0,contrast=1.0)
 a=frames_for(gray,still);b=frames_for(gray,moving);_,va=positions(still);_,vb=positions(moving)
 assert np.array_equal(a,b) and not np.array_equal(va[10:21],vb[10:21]) and stationary(a) and stationary(b)
 out=dict(passed=True,images_identical=True,first_window_true_motion_still=bool(np.any(va[10:21])),first_window_true_motion_moving=bool(np.any(vb[10:21])),
  gate_on_still=stationary(a),gate_on_moving=stationary(b),case='Однотонное изображение и движение окна по нему; входные кадры совпадают, но заданные смещения различны.',
  images_sha256=hashlib.sha256(a.tobytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
  scope='Иллюстрация неразличимости одного синтетического входа, без нового запуска FlyVis; не оценка частоты реальных ошибок.')
 (OUT/'observability_check.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
