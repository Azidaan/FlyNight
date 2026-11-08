import os,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from collections import Counter
from research_bbox_readout_calibration_20261108.common import *


def main(obs):
    import numpy as np
    phase();r=read(RUN/'comparison_report.json');p=read(RUN/'protocol.json');rows=r['rows'];checks=Counter()
    for video,fold in r['folds'].items():
        train=[v for v in rows if v['video']!=video];held=[v for v in rows if v['video']==video];assert video not in fold['train_videos']
        for source,t in fold['thresholds'].items():
            negative=[v['scores'][source] for v in train if v['kind']=='foreign' and v['scores'][source] is not None];positive=[v['scores'][source] for v in train if v['kind']=='positive' and v['scores'][source] is not None]
            wanted='off' if not negative or not positive else float(np.nextafter(max(negative),np.inf));wanted='off' if wanted=='off' or not any(v>=wanted for v in positive) else wanted;assert t==wanted
            if t!='off':assert all(v<t for v in negative)
            checks['independent_fold_thresholds']+=1
        for arm,cfg in fold['configs'].items():
            accepted=[];counts=Counter(N=sum(v['kind']!='unknown' for v in held))
            for v in held:
                value=v['scores'][cfg['source']];yes=cfg['threshold']!='off' and value is not None and value>=cfg['threshold']
                if yes:counts[v['kind']]+=1;accepted.append(dict(frame=v['frame'],old_tid=v['old_tid'],new_tid=v['new_tid'],kind=v['kind']))
            counts['J']=counts['positive']-counts['foreign'];assert dict(counts)==r['per_video'][arm][video];assert accepted==next(x['accepted'] for x in r['ledgers'] if x['video']==video and x['arm']==arm);checks['held_video_ledgers']+=1
    videos=p['fit_videos'];ix=np.random.default_rng(42).integers(0,len(videos),(20000,len(videos)))
    for key,value in r['comparisons'].items():
        a,b=key.split('-');d=np.array([r['per_video'][a][v]['J']-r['per_video'][b][v]['J'] for v in videos]);n=np.array([r['per_video'][a][v]['N'] for v in videos]);samples=100*d[ix].sum(1)/np.maximum(1,n[ix].sum(1));assert value['delta_J']==int(d.sum());assert np.allclose(value['ci95'],np.percentile(samples,[2.5,97.5]),atol=1e-12,rtol=0);assert np.allclose(value['ci_corrected'],np.percentile(samples,[2.5/20,100-2.5/20]),atol=1e-12,rtol=0);checks['independent_intervals']+=1
    for n,h in read(RUN/'execution_inputs.json')['sources'].items():assert sha(ROOT/n)==h
    phase();write(RUN/'independent_audit.json',dict(passed=True,checks=dict(checks),comparison_sha256=sha(RUN/'comparison_report.json'),no_threshold_fitting_on26_or20val=True,scope='Independent class-threshold and held-video accounting; no claim of independent scene data, causal object identity, or full tracker quality.'),new=True)


if __name__=='__main__':main(guard())
