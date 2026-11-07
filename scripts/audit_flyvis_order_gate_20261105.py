"""Independent implementation of order feature with shared stimulus and baselines."""
from pathlib import Path
import hashlib,json,sys,time
import numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/flyvis_order_gate_20261105';sys.path.insert(0,str(ROOT))
from scripts.flyvis_order_gate_20261105 import image,stimulus
from scripts.flyvis_local_motion_gate_20261105 import features
from scripts.flyvis_brightness_gate_20261105 import corrected
def main():
 p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'));r=json.loads((OUT/'report.json').read_text(encoding='utf-8'));assert r['protocol_sha256']==hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest();cv2.setNumThreads(2);imgs=[image(v) for v in p['videos']];verified=True;maximum=0.;vals=[];start=time.perf_counter()
 for row in r['rows']:
  f=stimulus(imgs[row['case']['scene']],row['case']);verified &= hashlib.sha256(f.tobytes()).hexdigest()==row['input_sha256'];mx=0.;amin=1.
  for t in range(1,11):
   a=f[t-1].astype(np.int16);b=f[t].astype(np.int16);mask=np.zeros((512,512),np.uint8);active=total=0
   for dy,dx in ((0,1),(1,0),(0,2),(2,0)):
    old=a[dy:,dx:]-a[:512-dy,:512-dx];new=b[dy:,dx:]-b[:512-dy,:512-dx];eligible=(np.abs(old)>3)&(np.abs(new)>3);inv=((old<0)&(new>0))|((old>0)&(new<0));mask[:512-dy,:512-dx]+=(eligible&inv);active+=int(eligible.sum());total+=eligible.size
   amin=min(amin,active/total)
   for y in range(0,512,32):
    for x in range(0,512,32):mx=max(mx,float(mask[y:y+32,x:x+32].sum())/(4*32*32))
  maximum=max(maximum,abs(mx-row['score']),abs(amin-row['minimum_active']));cand='uncertain' if amin<.01 else ('stationary' if mx<r['threshold'] else 'moving');verified &= cand==row['candidate'];old=features(f);med,_=corrected(f);g=old['global'];lc='uncertain' if g['minimum_texture']<=1 else ('stationary' if old['local_max_mean_difference']<p['local_threshold'] else 'moving');mc='uncertain' if g['minimum_texture']<=1 else ('stationary' if med<p['local_threshold'] else 'moving');gc='uncertain' if not g['observable'] else ('stationary' if g['maximum_displacement']<p['global_threshold'] else 'moving');verified &= lc==row['local'] and mc==row['median'] and gc==row['global_decision'];vals.append((row['case'],mx,amin))
 cal=[mx for c,mx,act in vals if c['scene']<4 and c['kind'] in ('still','light','shadow','soft_shadow') and act>=.01];verified &= bool(cal) and max(cal)+1e-9==r['threshold'];result=dict(passed=bool(verified and maximum<1e-12),exact_inputs_and_decisions=bool(verified),max_order_feature_difference=maximum,report_sha256=hashlib.sha256((OUT/'report.json').read_bytes()).hexdigest(),scope='Independent loop for candidate, but same stimulus generator and shared implementations for old baselines; same scenes, not independent dataset.',seconds=time.perf_counter()-start);(OUT/'independent_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(result));assert result['passed']
if __name__=='__main__':main()
