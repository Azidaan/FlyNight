"""Additional image-vs-geometry contrasts registered before full20policy outputs."""
import os,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_identity_assignment_run_20261108.common import *
from research_singleton_verify_20261108.common import guard
FULL=ROOT/'runs/bbox_normalized_signals_20261108/full'
RUN=ROOT/'runs/identity_evidence_20261108'


def register():
    assert not (FULL/'rollouts_complete.json').exists() and not (FULL/'metrics_complete.json').exists(),'Register before full20policy outcomes, not after inspecting quality.'
    p=read(FULL/'protocol.json');assert len(set(map(tuple,p['primary_comparisons'])))+2<=20
    write(RUN/'image_geometry_addendum.json',dict(registered=datetime.now(MSK).isoformat(),sources={'runs/bbox_normalized_signals_20261108/full/protocol.json':sha(FULL/'protocol.json'),'research_identity_evidence_20261108/image_geometry_comparison.py':sha(__file__)},comparisons=[['IMAGE','BBOX'],['IMAGE','SUPPORT']],family=20,primary_criteria_and_selection_unchanged=True,reason='Explicitly separate image-veto effect from the identical ordinary-geometry policy. These are additional contrasts from the same saved official metrics, not new evaluator runs. Total registered contrasts19<=20; no thresholds or eligibility changed.'),new=True)


def main(obs):
    import numpy as np
    assert read(FULL/'queue_result.json')['passed'];p=read(FULL/'protocol.json');add=read(RUN/'image_geometry_addendum.json')
    for n,h in add['sources'].items():assert sha(ROOT/n)==h,n
    videos=list(p['videos']);N=len(videos)
    with np.load(FULL/'bootstrap_counts.npz',allow_pickle=False) as z:counts=z['counts']
    idx=np.random.default_rng(42).integers(0,N,(20000,N));expected=np.asarray([np.bincount(row,minlength=N) for row in idx]);assert np.array_equal(counts,expected);weights=np.vstack([np.ones((1,N)),counts]);samples={};points={};report=read(FULL/'comparison_report.json')
    for arm in ['IMAGE','BBOX','SUPPORT']:
        rows=[read(FULL/'metrics'/arm/(v+'.json')) for v in videos];parts={k:np.asarray([r['components'][k] for r in rows],float) for k in ['TP','FP','FN','AA']};tp=weights@parts['TP'];fp=weights@parts['FP'];fn=weights@parts['FN'];aa=weights@parts['AA'];h=100*np.sqrt(tp/np.maximum(1,tp+fp+fn)*aa/np.maximum(1,tp)).mean(1);points[arm]=h[0];samples[arm]=h[1:];assert abs(h[0]-report['pooled'][arm]['HOTA'])<1e-9
    comparisons={}
    for a,b in add['comparisons']:
        delta=samples[a]-samples[b];comparisons[a+'-'+b]=dict(delta_pooled_HOTA=points[a]-points[b],ci95=np.percentile(delta,[2.5,97.5]).tolist(),ci_corrected=np.percentile(delta,[2.5/20,100-2.5/20]).tolist())
    for n,h in add['sources'].items():assert sha(ROOT/n)==h
    write(RUN/'image_geometry_comparison.json',dict(passed=True,full_report_sha256=sha(FULL/'comparison_report.json'),full_independent_audit_sha256=sha(FULL/'independent_audit.json'),registered_addendum_sha256=sha(RUN/'image_geometry_addendum.json'),comparisons=comparisons,new_evaluator_calls=0,selection_and_acceptance_unchanged=True),new=True)


if __name__=='__main__':
    if '--register' in sys.argv:register()
    else:main(guard())
