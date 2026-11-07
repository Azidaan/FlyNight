"""Independent CPU evaluator of saved raw decoder outputs and known synthetic truth."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/flyvis_t4t5_tensor_skill_probe_20261105';SRC=ROOT/'runs/flyvis_y1_latency_fp64_20261104';sys.path.insert(0,str(ROOT))
from scripts.flyvis_y1_latency_fp64_20261104 import positions
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'));r=json.loads((OUT/'run_result.json').read_text(encoding='utf-8'));assert p['evaluator_sha256']==sha(Path(__file__)) and r['passed'] and r['outputs_sha256']==sha(OUT/'outputs.npz') and r['protocol_sha256']==sha(OUT/'protocol.json');data=np.load(OUT/'outputs.npz');ref=np.load(SRC/'outputs'/(p['case']['id']+'.npz'))['flow'][0,1];mask=np.load(SRC/'geometry.npz')['mask'];assert set(data.files)==set(p['arms']);assert all(data[n].shape==(79,721,2) and np.isfinite(data[n]).all() for n in data.files);assert np.max(np.abs(data['reference']-ref))<=1e-5 and np.array_equal(data['reference'],data['identity_t4']);truth=positions(p['case'])[1].astype(float);reference=data['reference'][:,mask].mean(1);out={}
 for name in p['arms']:
  v=data[name][:,mask].mean(1);after=v[55:79];sign=after[:,p['case']['axis']]<0;hit=next((i for i in range(22) if sign[i:i+3].all()),None);out[name]=dict(after_epe=float(np.linalg.norm(after-truth[55:79],axis=1).mean()),after_correct_fraction=float(sign.mean()),latency_first_three=hit,mean_shift_from_reference=float(np.linalg.norm((v-reference)[1:79],axis=1).mean()),max_shift_from_reference=float(np.linalg.norm((v-reference)[1:79],axis=1).max()))
 assert out['identity_t4']['mean_shift_from_reference']==0.;result=dict(passed=True,protocol_sha256=sha(OUT/'protocol.json'),input_outputs_sha256=sha(OUT/'outputs.npz'),run_result_sha256=sha(OUT/'run_result.json'),arms=out,scope='Independent evaluation of saved decoder outputs; same stimulus and ground-truth generator as the original, no independent network rerun or additional video. One case does not establish effect distribution or improvement.')
 (OUT/'independent_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
