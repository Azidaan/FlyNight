"""Official TrackEval and missing per-video components, in an isolated CPU child."""
import os,sys,argparse,gzip
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from research_identity_assignment_run_20261108.common import *
from research_identity_assignment_run_20261108.full_retina import FULL


def main():
    p=argparse.ArgumentParser();p.add_argument('video');p.add_argument('arm');p.add_argument('--parent-report',required=True);a=p.parse_args()
    from research_identity_assignment_run_20261108.full_cpu import Observer
    from evaluation.hota_eval import evaluate_hota,VALID
    obs=Observer(a.parent_report);folder=FULL/'metrics'/a.arm;folder.mkdir(parents=True,exist_ok=True)
    marker=folder/(a.video+'.json');details=folder/(a.video+'.details.json.gz')
    if marker.exists():assert sha(details)==read(marker)['details_sha256'];return
    assert not details.exists()
    status('full_official_evaluation',video=a.video,arm=a.arm,guard=obs.read())
    metrics=evaluate_hota(FULL/'outputs'/a.arm,[a.video],split='val',details_path=details)
    if a.arm=='B0':
        old=read(FULL/'protocol.json')['videos'][a.video]['baseline_metrics']
        for k,v in old.items():assert abs(metrics[k]-v)<1e-9,(a.video,k,metrics[k],v)
    with gzip.open(details,'rt',encoding='utf-8') as f:data=json.load(f)[a.video]
    H=[data[c]['HOTA'] for c in VALID if c in data]
    tp=sum(np.array(v['HOTA_TP'],float) for v in H);fp=sum(np.array(v['HOTA_FP'],float) for v in H);fn=sum(np.array(v['HOTA_FN'],float) for v in H)
    aa=sum(np.array(v['AssA'],float)*np.array(v['HOTA_TP'],float) for v in H)
    hota=float(100*np.sqrt((tp/np.maximum(1,tp+fp+fn))*(aa/np.maximum(1,tp))).mean())
    assa=float(100*(aa/np.maximum(1,tp)).mean())
    assert abs(hota-metrics['HOTA'])<1e-9 and abs(assa-metrics['AssA'])<1e-9
    identities={k:sum(int(data[c]['Identity'][k]) for c in VALID if c in data) for k in ['IDTP','IDFP','IDFN']}
    idf1=100*identities['IDTP']/max(1.,identities['IDTP']+.5*identities['IDFP']+.5*identities['IDFN'])
    assert abs(idf1-metrics['IDF1'])<1e-9
    result=dict(passed=True,video=a.video,arm=a.arm,metrics=metrics,
        components=dict(TP=tp.tolist(),FP=fp.tolist(),FN=fn.tolist(),AA=aa.tolist(),**identities),
        details_sha256=sha(details),output_sha256=sha(FULL/'outputs'/a.arm/(a.video+'.json')),
        baseline_scalar_parity_checked=a.arm=='B0')
    write(marker,result,new=True)


if __name__=='__main__':main()
