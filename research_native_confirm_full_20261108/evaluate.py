"""Evaluate changed public identities; reuse exact ready outputs and official details."""
import os,sys,gzip,shutil
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_native_confirm_full_20261108.common import *


def main(obs):
    import numpy as np
    from evaluation.hota_eval import evaluate_hota,VALID
    full_phase();p=read(FULL/'protocol.json');assert read(FULL/'rollouts_complete.json')['passed'];cache={};reused=0;new=0;manifest={}
    for video in p['videos']:
        for arm in p['parameters']:
            folder=FULL/'metrics'/arm;folder.mkdir(parents=True,exist_ok=True);dest=folder/(video+'.json');details=folder/(video+'.details.json.gz');output=FULL/'outputs'/arm/(video+'.json');signature=(video,sha(output))
            if dest.exists():
                r=read(dest);assert sha(details)==r['details_sha256'] and sha(output)==r['output_sha256'];cache[signature]=dest;manifest[f'{arm}/{video}']=sha(dest);continue
            candidate=cache.get(signature)
            if candidate is None:
                candidate=next((ROOT/e['path'] for e in p['ready_metric_sources'][video] if e['output_sha256']==signature[1]),None)
            if candidate is not None:
                r=read(candidate);old_details=candidate.with_suffix('.details.json.gz');assert sha(old_details)==r['details_sha256'];assert r['output_sha256']==signature[1];shutil.copyfile(old_details,details)
                r=dict(r,arm=arm,reused_metric_source=candidate.relative_to(ROOT).as_posix(),identical_output_json_verified=True);write(dest,r,new=True);reused+=1
            else:
                metrics=evaluate_hota(FULL/'outputs'/arm,[video],split='val',details_path=details)
                with gzip.open(details,'rt',encoding='utf-8') as f:d=json.load(f)[video]
                H=[d[c]['HOTA'] for c in VALID if c in d];tp=sum(np.asarray(h['HOTA_TP'],float) for h in H);fp=sum(np.asarray(h['HOTA_FP'],float) for h in H);fn=sum(np.asarray(h['HOTA_FN'],float) for h in H);aa=sum(np.asarray(h['AssA'],float)*np.asarray(h['HOTA_TP'],float) for h in H)
                assert abs(float(100*np.sqrt(tp/np.maximum(1,tp+fp+fn)*aa/np.maximum(1,tp)).mean())-metrics['HOTA'])<1e-9
                identities={k:sum(int(d[c]['Identity'][k]) for c in VALID if c in d) for k in ['IDTP','IDFP','IDFN']}
                r=dict(passed=True,video=video,arm=arm,metrics=metrics,components=dict(TP=tp.tolist(),FP=fp.tolist(),FN=fn.tolist(),AA=aa.tolist(),**identities),details_sha256=sha(details),output_sha256=signature[1]);write(dest,r,new=True);new+=1
            cache[signature]=dest;manifest[f'{arm}/{video}']=sha(dest);status('native_confirm_full_official_metrics',video=video,arm=arm,reused=reused,new=new,total=len(p['videos'])*len(p['parameters']),guard=obs.read())
    full_phase();write(FULL/'metrics_complete.json',dict(passed=True,reused=reused,new=new,metrics=manifest),new=True)


if __name__=='__main__':main(guard())
