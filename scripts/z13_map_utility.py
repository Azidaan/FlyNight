"""CPU response-map features on fixed initial D/C/R1; video-held-out utility."""
from datetime import datetime,timezone
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.z13_parallel_next import read,write,sha
RUN=ROOT/'runs/z13_map_utility_20261018'
DEADLINE=None


def status(**values):
    if time.time()>=DEADLINE or (RUN/'stop.request').exists():
        raise TimeoutError('Authorized CPU budget ended')
    write(RUN/'status.json',dict(state='running',updated=datetime.now(timezone.utc).isoformat(),**values))


def verify(protocol):
    for path,digest in protocol['sha256'].items():
        if sha(ROOT/path)!=digest:
            raise RuntimeError('Frozen source changed: '+path)


def main():
    global DEADLINE
    protocol=read(RUN/'protocol.json');verify(protocol)
    DEADLINE=datetime.fromisoformat(protocol['save_stop_at']).timestamp()
    if (RUN/'report.json').exists():
        raise RuntimeError('Do not repeat a completed run')
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['PYTORCH_NVML_BASED_CUDA_CHECK']='0'
    import torch
    torch.cuda.is_available=lambda:False;torch.set_default_device('cpu');torch.set_num_threads(2)
    from scripts import z13_target_model as old
    from scripts import z13_target_center_probe as capture
    from scripts.z13_train import Episodes
    from tracking import z13_target_center as original
    from tracking.z13_response_utility import response_features,NAMES
    from tracking.z13_utility import observable_features,leave_video_out
    from scripts.z13_continue import coverage_table,per_video_useful,paired
    from scripts.z13_score_swap import key,selected_keys
    checkpoint=torch.load(ROOT/'runs/z13_spatial_v2_20261013/D/full/model.pt',map_location='cpu',weights_only=False)
    choices=read(ROOT/'runs/z13_target_center_20261018/report.json')['choices']
    fit=Episodes(protocol['fit20']);metadata={};offset=0
    for video in protocol['videos']:
        with np.load(ROOT/f'runs/z13_data/{video}.npz') as data:
            for i,crop in enumerate(data['lab_crop']):
                metadata[(video,str(offset+i))]=crop.copy()
            offset+=len(data['lab_kind'])
    results={};gated={}
    for arm in ('D','C','R1'):
        folder=RUN/arm;folder.mkdir(exist_ok=True)
        reference=read(ROOT/f'runs/z13_target_center_20261018/{arm}/center_rows.json')
        references={key(r):r for r in reference}
        if (folder/'done.json').exists():
            results[arm]=read(folder/'done.json')
            saved={(r['video'],str(r['label'])):r['score'] for r in read(folder/'predicted_scores.json')}
            out=[dict(r,p_present=saved[r['video'],str(r['label'])]) for r in reference]
            chosen=selected_keys(out,.01)
            gated[arm]=per_video_useful([dict(r,useful=bool(r.get('useful')) and key(r) in chosen) for r in out])
            continue
        if not (folder/'feature_index.json').exists():
            status(stage='feature normalization',variant=arm,done=0,total=200)
            bank,geom,bb=old.build_arm(arm,torch,checkpoint)
            assert bb is None or str(bb.dev)=='cpu'
            if (folder/'stats.npz').exists():
                with np.load(folder/'stats.npz') as z: stats=(z['mu'],z['sd'])
            else:
                stats=old.feature_stats(torch,arm,bb,geom,fit)
                np.savez(folder/'stats.npz',mu=stats[0],sd=stats[1])
            all_features={};offset=0
            for vi,video in enumerate(protocol['videos']):
                output=folder/(video+'.json')
                with np.load(ROOT/f'runs/z13_data/{video}.npz') as z:
                    label_count=len(z['lab_kind'])
                if output.exists():
                    for item in read(output)['features']:
                        all_features[(video,str(item['label']))]=item['values']
                    offset+=label_count;continue
                data=Episodes([video]);captured=[]
                original_function=original.ridge_scores
                def tapping(*args,**kwargs):
                    scores=original_function(*args,**kwargs)
                    assert scores.shape[0]==1
                    captured.extend(response_features(score,geom).tolist() for score in scores[0])
                    return scores
                def progress(**kwargs):
                    status(**kwargs,variant=arm,video=video,videos_done=vi,videos_total=len(protocol['videos']))
                old_modes=capture.MODES;old_status=capture.status
                capture.MODES=('center',);capture.status=progress;original.ridge_scores=tapping
                try:
                    rows,_=capture.paired_rows(torch,arm,bb,geom,bank.eye_size,data,
                        {'center':[choices[arm]['center']['lambda_value']]},stats,'unchanged map capture')
                finally:
                    capture.MODES=old_modes;capture.status=old_status;original.ridge_scores=original_function
                rows=old.useful_rows(rows['center'],0)
                assert len(rows)==len(captured)
                dc=dp=0.;features=[]
                for row,values in zip(rows,captured):
                    row=dict(row,label=row['label']+offset,video=video)
                    ref=references[key(row)]
                    dc=max(dc,float(np.max(np.abs(np.asarray(row['center'])-ref['center']))))
                    dp=max(dp,abs(row['p_present']-ref['p_present']))
                    assert bool(row.get('useful'))==bool(ref.get('useful'))
                    all_features[(video,str(row['label']))]=values
                    features.append(dict(label=row['label'],values=values))
                assert dc<.5 and dp<.001,(arm,video,dc,dp)
                write(output,dict(features=features,max_center_difference=dc,max_peak_difference=dp,
                                  known_outcomes_match=True))
                offset+=label_count
            assert set(all_features)=={(r['video'],str(r['label'])) for r in reference}
            write(folder/'feature_index.json',dict(names=NAMES,rows=len(all_features),videos=len(protocol['videos'])))
            del bb,bank
        else:
            all_features={}
            for video in protocol['videos']:
                for item in read(folder/(video+'.json'))['features']:
                    all_features[(video,str(item['label']))]=item['values']
        x=[]
        for row in reference:
            pair=(row['video'],str(row['label']))
            geometry,_=observable_features(row,metadata[pair])
            x.append(np.r_[geometry,all_features[pair]])
        targets=[int(bool(r.get('useful'))) if r['kind']>=0 else -1 for r in reference]
        def on_fold(done,total,video):
            status(stage='full-map utility fitting',variant=arm,done=done,total=total,video=video)
        scores,folds=leave_video_out(x,targets,[r['video'] for r in reference],on_fold)
        limited=read(ROOT/f'runs/z13_parallel_next_20261018/utility/{arm}_rows.json')
        variants=dict(original=reference,geometry=limited['geometry'],limited=limited['utility'],
                      map_utility=[dict(r,p_present=float(score)) for r,score in zip(reference,scores)])
        summary={name:coverage_table(rows) for name,rows in variants.items()};comparisons={};pv={}
        for q in (.01,.02,.05):
            for name,rows in variants.items():
                chosen=selected_keys(rows,q)
                pv[name,q]=per_video_useful([dict(r,useful=bool(r.get('useful')) and key(r) in chosen) for r in rows])
            comparisons[f'{q:.2f}']={'map_utility-'+baseline:paired(pv['map_utility',q],pv[baseline,q])
                                    for baseline in ('original','geometry','limited')}
        results[arm]=dict(coverage=summary,paired_selected_success=comparisons)
        gated[arm]=pv['map_utility',.01]
        write(folder/'predicted_scores.json',[dict(label=r['label'],video=r['video'],score=float(s)) for r,s in zip(reference,scores)])
        write(folder/'folds.json',folds);write(folder/'done.json',results[arm])
    primary=all(item['ci95'][0]>0 for item in results['D']['paired_selected_success']['0.01'].values())
    cross={f'D_map_utility-{arm}_map_utility':paired(gated['D'],gated[arm]) for arm in ('C','R1')}
    assert not torch.cuda.is_initialized();verify(protocol)
    write(RUN/'report.json',dict(arms=results,paired=cross,primary_pass=primary,
          fly_control_pass=cross['D_map_utility-R1_map_utility']['ci95'][0]>0,
          decision='candidate_for_separate_validation' if primary else 'full_map_utility_gate_not_passed',
          cuda_initialized=False,dev_evaluated=False,hota_measured=False,limitations=protocol['limitations']))
    write(RUN/'status.json',dict(state='completed',stage='report saved',done=1,total=1,updated=datetime.now(timezone.utc).isoformat()))


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        write(RUN/'status.json',dict(state='paused' if isinstance(exc,TimeoutError) else 'failed',stage=str(exc),done=0,total=0,
                                   updated=datetime.now(timezone.utc).isoformat()))
        raise
