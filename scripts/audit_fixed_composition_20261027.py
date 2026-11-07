"""Независимо восстановить все фиксированные координаты и парные интервалы."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,copy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.recovery_research_service_20261027 import RUN as BASE,read,write,sha,matching,now

def main():
    import numpy as np
    from tracking.fixed_box_calibration import calibrate_rows
    from tracking.pixel_box_refinement import coordinates,metadata,apply_offsets
    run=BASE/'fixed_composition';p=read(run/'protocol.json');r=read(run/'report.json');c=read(run/'controller_result.json')
    assert r['complete'] and c['exit_code']==0 and c['all_owned_ended'] and not any(matching(k,v) for k,v in c['owned'].items())
    for f,h in p['sha256'].items():assert sha(ROOT/f)==h,f
    fit=np.load(BASE/'pixel_box_numeric_control/calibration.npz');checked=0;strict=True
    for s in p['sources']:
        v=s['video'];frames=[int(x['video_frame_index']) for x in read(ROOT/s['alignment'])['matches']]
        names=[('baseline','numeric',ROOT/s['baseline']),('appearance','appearance_numeric',BASE/'lazy_appearance_validation/appearance'/(v+'.json'))]
        if s.get('masa'):names.append(('masa','masa_numeric',ROOT/s['masa']))
        for original,adjusted,file in names:
            rows=read(file);assert rows==read(run/original/(v+'.json'))
            zero,_,_=calibrate_rows(rows,frames,fit,0);assert zero==rows
            expected=copy.deepcopy(rows);past={}
            for ref,(row,frame) in enumerate(zip(rows,frames)):
                for j,label in enumerate(row['labels']):
                    old=past.get(str(label['id']));assert old is None or old<frame
                    m=metadata(label,old is not None,0 if old is None else frame-old).astype(np.float64)
                    d=np.clip(np.r_[1,np.clip((m-fit['mean'])/fit['std'],-10,10)]@fit['beta'],-.25,.25).astype(np.float32)
                    if np.any(d):expected[ref]['labels'][j]['box2d']={k:float(x) for k,x in zip(['x1','y1','x2','y2'],apply_offsets(coordinates(label),d,.5))}
                past={str(l['id']):frame for l in row['labels']}
            actual=read(run/adjusted/(v+'.json'));assert actual==expected,(v,adjusted)
            if original=='baseline':assert actual==read(BASE/'pixel_box_numeric_control/numeric_control'/(v+'.json'))
            checked+=len(rows)
        write(BASE/'fixed_composition_audit/status.json',dict(state='running',updated=now(),video=v,checked_output_frames=checked),required=False)
    assert set(r['per_video']['masa'])==set(p['groups']['development20'])
    old=read(ROOT/'runs/z13_val20_matched_20261024/native_masa_recovery2/report.json')
    # Число сильного контроля сверяется с неизменным старым первичным отчётом.
    assert abs(r['pooled']['masa']['HOTA']-old['pooled']['xywh_masa']['HOTA'])<1e-8
    assert set(old['videos'])==set(p['groups']['development20'])
    for v in old['videos']:
        assert r['per_video']['masa'][v]==old['per_video']['xywh_masa'][v]
    pairs={};rng=np.random.default_rng(281031)
    for group,comparisons in r['paired'].items():
        vs=[s['video'] for s in p['sources']] if group=='all40' else p['groups'][group];pairs[group]={}
        for key,value in comparisons.items():
            aa,bb=key.split('_vs_');d=np.array([r['per_video'][aa][v]['HOTA']-r['per_video'][bb][v]['HOTA'] for v in vs])
            assert abs(d.mean()-value['mean_per_video'])<1e-10 and value['n']==len(vs)
            boot=d[rng.integers(0,len(vs),(5000,len(vs)))].mean(1);pairs[group][key]=dict(mean_per_video=float(d.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),n=len(vs))
    passed=all(x['ci95'][0]>0 for x in r['paired']['all40'].values())
    audit=dict(passed=True,complete_videos=40,masa_videos=20,all_outputs_reconstructed=True,zero_exact=True,old_numeric_exact=True,strict_past=True,
        independent_pairs=pairs,prespecified_composition_better_than_both_parents=passed,source_hashes_verified=True,report_sha256=sha(run/'report.json'),updated=now())
    write(run/'independent_audit.json',audit);write(BASE/'fixed_composition_audit/report.json',audit);print('All fixed composition outputs audited')

if __name__=='__main__':main()
