"""Независимая сверка числовой калибровки, всех JSON и парных контрастов."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,copy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.recovery_research_service_20261027 import RUN as BASE,read,write,sha,matching,now
RUN=BASE/'pixel_box_numeric_control'

def main():
    import numpy as np
    from tracking.pixel_box_refinement import metadata,apply_offsets
    p=read(RUN/'protocol.json');r=read(RUN/'report.json');c=read(RUN/'controller_result.json')
    assert r['complete'] and c['exit_code']==0 and c['all_owned_ended'] and not any(matching(k,v) for k,v in c['owned'].items())
    for path,h in p['sha256'].items():assert sha(ROOT/path)==h,path
    f=read(RUN/'frozen_before_validation.json');assert f==r['frozen'] and f['calibration_sha256']==sha(RUN/'calibration.npz')
    fit=np.load(RUN/'calibration.npz');dev=read(RUN/'dev_report.json');candidates=[k for k in dev['mean_per_video'] if k!='baseline']
    best=max(candidates,key=lambda n:(dev['mean_per_video'][n],-float(n.rsplit('_a',1)[1]),n=='class_median_a0.0'));assert best==f['name']
    for source in p['sources']:
        v=source['video'];base=read(ROOT/source['baseline']);assert base==read(RUN/'baseline'/(v+'.json'));out=copy.deepcopy(base)
        lookup=[{str(x['id']):x for x in row['labels']} for row in out];records=read(BASE/'pixel_box_validation/offsets'/(v+'_records.json'));delta=np.load(RUN/(v+'_offsets.npy'))
        assert sha(RUN/(v+'_offsets.npy'))==r['proof'][v]['offsets_sha256'] and len(delta)==len(records)
        for rec,d in zip(records,delta):
            label=lookup[rec['reference']][str(rec['id'])];m=metadata(label,rec['past_frame'] is not None,0 if rec['past_frame'] is None else rec['frame']-rec['past_frame']).astype(np.float64)
            if f['method']=='class_median':expected=fit['class_medians'][np.argmax(m[4:])]
            else:expected=np.clip(np.r_[1,np.clip((m-fit['mean'])/fit['std'],-10,10)]@fit['beta'],-.25,.25)
            assert np.allclose(d,expected,atol=1e-7,rtol=1e-6)
            if f['alpha'] and np.any(d):label['box2d']={k:float(x) for k,x in zip(['x1','y1','x2','y2'],apply_offsets(rec['box'],d,f['alpha']))}
        assert read(RUN/'numeric_control'/(v+'.json'))==out
        assert r['per_video']['baseline'][v]==source['baseline_metrics']
    visual=read(BASE/'pixel_box_validation/report.json');independent={};rng=np.random.default_rng(281029);decision={}
    for group,contrasts in r['paired'].items():
        vs=[s['video'] for s in p['sources']] if group=='all40' else p['groups'][group];independent[group]={}
        for key,value in contrasts.items():
            if key=='numeric_vs_baseline':aa=r['per_video']['numeric_control'];bb=r['per_video']['baseline']
            else:aa=visual['per_video'][key.removesuffix('_vs_numeric')];bb=r['per_video']['numeric_control']
            d=np.array([aa[v]['HOTA']-bb[v]['HOTA'] for v in vs]);assert abs(d.mean()-value['mean_per_video'])<1e-10 and value['n']==len(vs)
            boot=d[rng.integers(0,len(vs),(5000,len(vs)))].mean(1);independent[group][key]=dict(mean_per_video=float(d.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),n=len(vs))
        for arm in ['current','temporal']:decision[group+'_'+arm+'_all3_vs_numeric']=all(contrasts[f'{arm}_seed{s}_a0.5_vs_numeric']['ci95'][0]>0 for s in [42,43,44])
    audit=dict(passed=True,complete_videos=40,all_offsets_and_outputs_reconstructed=True,weights_frozen_on_dev=True,source_hashes_verified=True,
        independent_pairs=independent,prespecified_decision=decision,report_sha256=sha(RUN/'report.json'),updated=now())
    write(RUN/'independent_audit.json',audit);write(BASE/'pixel_box_numeric_audit/report.json',audit);print('Calibration audited: '+str(decision))

if __name__=='__main__':main()
