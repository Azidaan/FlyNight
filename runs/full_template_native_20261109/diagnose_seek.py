"""Discriminate MOV seek mismatch from checkpoint-state mismatch."""
from common import *
start('cpu')
import cv2
import numpy as np
cv2.setNumThreads(2)
p=read(OUT/'protocol.json');verify(p);w=p['witnesses'][0];entry=p['videos'][w['video']];source=ROOT/entry['video_path']
assert sha(source)==entry['sources'][entry['video_path']]
write('seek_diagnostic_protocol.json',dict(registered=datetime.now(MSK).isoformat(),scope='Raw decoded frames100 and117 only; no retina,model,matching or score.',
    sources={Path(__file__).relative_to(ROOT).as_posix():sha(Path(__file__)),entry['video_path']:sha(source),'runs/full_template_native_20261109/protocol.json':sha(OUT/'protocol.json')}))
checkpoints=sorted((LEGACY/'processed'/(w['video']+'.checkpoints')).glob('*.json'));start_frame=max(read(path)['next_frame'] for path in checkpoints if read(path)['next_frame']<=w['frame'])
a=cv2.VideoCapture(str(source));b=cv2.VideoCapture(str(source));assert a.isOpened() and b.isOpened();assert b.set(cv2.CAP_PROP_POS_FRAMES,start_frame)
records=[]
try:
    for frame in range(w['frame']+1):
        ok,x=a.read();assert ok
        if frame<start_frame:continue
        ok,y=b.read();assert ok
        if frame not in [start_frame,w['frame']]:continue
        records.append(dict(frame=frame,reported_seek_position=b.get(cv2.CAP_PROP_POS_FRAMES),exact=np.array_equal(x,y),mean_abs_pixel_difference=float(np.abs(x.astype(float)-y.astype(float)).mean()),max_abs_pixel_difference=int(np.abs(x.astype(int)-y.astype(int)).max()),sequential_sha256=hashlib.sha256(x.tobytes()).hexdigest(),seek_sha256=hashlib.sha256(y.tobytes()).hexdigest()))
finally:a.release();b.release()
write('seek_diagnostic.json',dict(passed=True,checkpoint_start=start_frame,records=records,any_raw_seek_mismatch=any(not r['exact'] for r in records),no_retina_or_model_calls=True))
print(json.dumps(records),flush=True)
