"""Independently recalculate features and decisions using the same stimulus generator."""
from pathlib import Path
import hashlib,json,sys,time
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/flyvis_local_gate_stress_20261105';sys.path.insert(0,str(ROOT))
from scripts.flyvis_local_gate_stress_20261105 import image,make
def main():
 p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'));r=json.loads((OUT/'report.json').read_text(encoding='utf-8'));assert r['protocol_sha256']==hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest();cv2.setNumThreads(2);imgs=[image(v) for v in p['videos']];max_local=max_global=0.;valid=True;start=time.perf_counter()
 for row in r['rows']:
  frames=make(imgs[row['case']['scene']],row['case']);valid &= hashlib.sha256(frames.tobytes()).hexdigest()==row['input_sha256'];local=0.;shifts=[];quality=[];texture=[]
  for i in range(1,11):
   x=frames[i-1].astype(np.float32);y=frames[i].astype(np.float32);d=np.abs(x-y)
   for u in range(0,512,32):
    for v in range(0,512,32):local=max(local,float(d[u:u+32,v:v+32].mean()))
   shift,q=cv2.phaseCorrelate(x,y);shifts.append(float(np.hypot(*shift)));quality.append(float(q));texture.append(min(float(x.std()),float(y.std())))
  g=row['features']['global'];max_local=max(max_local,abs(local-row['features']['local_max_mean_difference']));max_global=max(max_global,abs(max(shifts)-g['maximum_displacement']));ld='uncertain' if min(texture)<=1 else ('stationary' if local<p['local_threshold'] else 'moving');gd='uncertain' if min(quality)<.2 or min(texture)<=1 else ('stationary' if max(shifts)<p['global_threshold'] else 'moving');valid &= ld==row['local'] and gd==row['global_decision']
 passed=bool(valid and max_local<1e-5 and max_global<1e-8);result=dict(passed=passed,all_inputs_and_decisions_exact=bool(valid),max_local_diff=max_local,max_global_diff=max_global,report_sha256=hashlib.sha256((OUT/'report.json').read_bytes()).hexdigest(),scope='Independently calculated features and decisions, same stimulus-generation code and scenes; not a new independent dataset.',seconds=time.perf_counter()-start);(OUT/'independent_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(result));assert passed
if __name__=='__main__':main()
