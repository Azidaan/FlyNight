"""Read-only reconstruction of saturation in already saved brightness cases."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/flyvis_brightness_gate_20261105';sys.path.insert(0,str(ROOT))
from scripts.flyvis_brightness_gate_20261105 import image,stimulus
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'));r=json.loads((OUT/'report.json').read_text(encoding='utf-8'));imgs=[image(v) for v in p['videos']];rows=[]
 for q in r['rows']:
  if q['case']['kind']!='light':continue
  x=stimulus(imgs[q['case']['scene']],q['case']);assert hashlib.sha256(x.tobytes()).hexdigest()==q['input_sha256'];rows.append(dict(case=q['case'],decision=q['candidate'],fraction_at_255_last_frame=float(np.mean(x[-1]==255)),input_sha256=q['input_sha256']))
 groups={}
 for noise in (0.,1.):
  for dec in ('stationary','moving'):
   z=[x['fraction_at_255_last_frame'] for x in rows if x['case']['noise']==noise and x['decision']==dec]
   if z:groups[f'noise_{noise}_{dec}']=dict(count=len(z),mean=float(np.mean(z)),min=float(min(z)),max=float(max(z)))
 doc=dict(source_report_sha256=sha(OUT/'report.json'),generator_sha256=sha(ROOT/'scripts/flyvis_brightness_gate_20261105.py'),review_script_sha256=sha(Path(__file__)),all_inputs_reconstructed=True,groups=groups,rows=rows,interpretation='The simulated light increment is additive followed by clipping; clipped pixels occur and correlate descriptively with decisions in these cases. This is not a controlled causal proof of the exact reason for every error, and it is not a new independent test.')
 target=OUT/'gemini_saturation_review.json';assert not target.exists();target.write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(groups))
if __name__=='__main__':main()
