"""Rebuild cases and independently calculate local/global features for the diagnostic."""
from pathlib import Path
import hashlib,json,sys,time
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/flyvis_local_motion_gate_20261105';sys.path.insert(0,str(ROOT))
from scripts.flyvis_local_motion_gate_20261105 import image,frames
def main():
 p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'));r=json.loads((OUT/'report.json').read_text(encoding='utf-8'));assert r['protocol_sha256']==hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest();cv2.setNumThreads(2);images=[image(x) for x in p['videos']];max_local=max_global=0.;all_input=True;all_decision=True;start=time.perf_counter()
 for row in r['rows']:
  a=frames(images[row['case']['scene']],row['case']);all_input &= hashlib.sha256(a.tobytes()).hexdigest()==row['input_sha256'];local=0.; shifts=[];response=[];texture=[]
  for t in range(1,11):
   first=a[t-1].astype(np.float32);second=a[t].astype(np.float32);delta=np.abs(second-first)
   for y in range(0,512,32):
    for x in range(0,512,32):local=max(local,float(delta[y:y+32,x:x+32].mean()))
   shift,q=cv2.phaseCorrelate(first,second);shifts.append(float(np.hypot(*shift)));response.append(float(q));texture.append(min(float(first.std()),float(second.std())))
  gf=row['features']['global'];max_local=max(max_local,abs(local-row['features']['local_max_mean_difference']));max_global=max(max_global,abs(max(shifts)-gf['maximum_displacement']));ld='uncertain' if min(texture)<=1 else ('stationary' if local<r['threshold'] else 'moving');gd='uncertain' if min(response)<.2 or min(texture)<=1 else ('stationary' if max(shifts)<p['global_threshold'] else 'moving');all_decision &= ld==row['local'] and gd==row['global']
 cal=max(x['features']['local_max_mean_difference'] for x in r['rows'] if x['case']['scene']<4 and x['case']['kind']=='still')+1e-6; passed=all_input and all_decision and max_local<1e-5 and max_global<1e-8 and cal==r['threshold'];out=dict(passed=passed,all_inputs_exact=bool(all_input),all_decisions_exact=bool(all_decision),max_local_diff=max_local,max_global_diff=max_global,threshold_recomputed=cal,report_sha256=hashlib.sha256((OUT/'report.json').read_bytes()).hexdigest(),scope='Independent feature and decision calculations; same deterministic stimulus generation and same scenes, not an independent dataset.',seconds=time.perf_counter()-start);(OUT/'independent_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(out));assert passed
if __name__=='__main__':main()
