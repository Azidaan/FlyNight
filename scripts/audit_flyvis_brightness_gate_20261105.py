"""Recompute local and global decisions with independent feature loops."""
from pathlib import Path
import hashlib,json,sys,time
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/flyvis_brightness_gate_20261105';sys.path.insert(0,str(ROOT))
from scripts.flyvis_brightness_gate_20261105 import image,stimulus
def main():
 p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'));r=json.loads((OUT/'report.json').read_text(encoding='utf-8'));assert r['protocol_sha256']==hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest();cv2.setNumThreads(2);imgs=[image(v) for v in p['videos']];valid=True;maxdiff=0.;start=time.perf_counter();counts={}
 for row in r['rows']:
  fs=stimulus(imgs[row['case']['scene']],row['case']);valid &= hashlib.sha256(fs.tobytes()).hexdigest()==row['input_sha256'];lc=pc=0.;qs=[];shifts=[];textures=[]
  for t in range(1,11):
   a=fs[t-1].astype(np.float32);b=fs[t].astype(np.float32);delta=b-a;med=float(np.median(delta));q=np.abs(delta-med);old=np.abs(delta)
   for i in range(0,512,32):
    for j in range(0,512,32):lc=max(lc,float(q[i:i+32,j:j+32].mean()));pc=max(pc,float(old[i:i+32,j:j+32].mean()))
   shift,z=cv2.phaseCorrelate(a,b);shifts.append(float(np.hypot(*shift)));qs.append(float(z));textures.append(min(float(a.std()),float(b.std())))
  maxdiff=max(maxdiff,abs(lc-row['corrected_max']),abs(pc-row['previous_features']['local_max_mean_difference']));decision='uncertain' if min(textures)<=1 else ('stationary' if lc<p['threshold'] else 'moving');old='uncertain' if min(textures)<=1 else ('stationary' if pc<p['threshold'] else 'moving');glob='uncertain' if min(qs)<.2 or min(textures)<=1 else ('stationary' if max(shifts)<p['global_threshold'] else 'moving');valid &= decision==row['candidate'] and old==row['local'] and glob==row['global_decision'];k=row['case']['kind'];counts.setdefault(k,{});counts[k][decision]=counts[k].get(decision,0)+1
 for k,c in counts.items():
  for decision in ('stationary','moving','uncertain'):valid &= c.get(decision,0)==r['summaries'][k]['candidate_'+decision]
 result=dict(passed=bool(valid and maxdiff<1e-5),all_inputs_and_decisions_exact=bool(valid),max_local_difference=maxdiff,report_sha256=hashlib.sha256((OUT/'report.json').read_bytes()).hexdigest(),scope='Independent feature and decision loops; same deterministic stimulus generator and scenes.',seconds=time.perf_counter()-start);(OUT/'independent_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(result));assert result['passed']
if __name__=='__main__':main()
