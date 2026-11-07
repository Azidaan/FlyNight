"""Фиксированная композиция назначений/координат и сильный старый MASA-контроль, только CPU."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
from datetime import datetime
import sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.recovery_research_service_20261027 import RUN as BASE,read,write,sha,matching,now

def main():
    import numpy as np
    from tracking.fixed_box_calibration import calibrate_rows
    from evaluation.hota_eval import evaluate_hota
    from scripts.z13_continue import paired
    run=BASE/'fixed_composition';p=read(run/'protocol.json');end=datetime.fromisoformat(read(run/'execution_deadline.json')['deadline']).timestamp()
    assert read(BASE/'lazy_appearance_validation/independent_audit.json')['passed']
    for folder in ['lazy_appearance_validation','lazy_appearance_validation_audit']:
        c=read(BASE/folder/'controller_result.json');assert c['exit_code']==0 and c['all_owned_ended'] and not any(matching(k,v) for k,v in c['owned'].items())
    for f,h in p['sha256'].items():assert sha(ROOT/f)==h,f
    frozen=read(BASE/'pixel_box_numeric_control/frozen_before_validation.json');assert frozen['method']=='ridge_meta' and frozen['alpha']==.5
    fit=np.load(BASE/'pixel_box_numeric_control/calibration.npz');old=read(BASE/'lazy_appearance_validation/report.json');numeric=read(BASE/'pixel_box_numeric_control/report.json')
    per={k:{} for k in ['baseline','numeric','appearance','appearance_numeric','masa','masa_numeric']};timing={};proof={};start=time.monotonic()
    def status(**kw):
        if time.time()>end-5 or (run/'stop.request').exists():raise TimeoutError('Fixed composition original budget')
        write(run/'status.json',dict(state='running',updated=now(),**kw),required=False)
    for i,s in enumerate(p['sources']):
        began=time.monotonic();v=s['video'];frames=[int(x['video_frame_index']) for x in read(ROOT/s['alignment'])['matches']]
        b=read(ROOT/s['baseline']);a=read(BASE/'lazy_appearance_validation/appearance'/(v+'.json'))
        nb,br,bd=calibrate_rows(b,frames,fit,.5);assert nb==read(BASE/'pixel_box_numeric_control/numeric_control'/(v+'.json'))
        an,ar,ad=calibrate_rows(a,frames,fit,.5)
        values=dict(baseline=b,numeric=nb,appearance=a,appearance_numeric=an)
        if s.get('masa'):
            masa=read(ROOT/s['masa']);assert len(masa)==len(b)
            # Сопоставимый старый MASA менял только ID, а не детекции/рамки.
            for x,y in zip(masa,b):
                assert {k:z for k,z in x.items() if k!='labels'}=={k:z for k,z in y.items() if k!='labels'}
                assert len(x['labels'])==len(y['labels'])
                assert all({k:z for k,z in xx.items() if k!='id'}=={k:z for k,z in yy.items() if k!='id'} for xx,yy in zip(x['labels'],y['labels']))
            mn,mr,md=calibrate_rows(masa,frames,fit,.5);values.update(masa=masa,masa_numeric=mn)
        for name,rows in values.items():
            status(video=v,index=i,total=40,variant=name);write(run/name/(v+'.json'),rows)
            if name=='baseline':metric=old['per_video']['baseline'][v]
            elif name=='numeric':metric=numeric['per_video']['numeric_control'][v]
            elif name=='appearance':metric=old['per_video']['appearance'][v]
            else:metric=evaluate_hota(run/name,[v],split='val')
            per[name][v]=metric
        proof[v]=dict(exact_old_numeric=True,only_composed_boxes_changed=True,causal_metadata=True)
        timing[v]=time.monotonic()-began;write(run/'per_video.json',per)
        if i==0:
            cost=2*timing[v]*(len(p['sources'])-1)+180;available=end-time.time()
            write(run/'price.json',dict(first_full_video_seconds=timing[v],conservative_remaining_seconds=cost,available_seconds=available,proceed=cost<available));assert cost<available
    pooled={name:evaluate_hota(run/name,list(values),split='val') for name,values in per.items()};groups=dict(p['groups']);groups['all40']=[s['video'] for s in p['sources']];pairs={}
    for group,vs in groups.items():
        comparisons=[('appearance_numeric','baseline'),('appearance_numeric','appearance'),('appearance_numeric','numeric')]
        if group=='development20':comparisons += [('appearance','masa'),('appearance_numeric','masa'),('masa_numeric','masa')]
        pairs[group]={aa+'_vs_'+bb:paired({v:per[aa][v]['HOTA'] for v in vs},{v:per[bb][v]['HOTA'] for v in vs}) for aa,bb in comparisons}
    write(run/'report.json',dict(complete=True,per_video=per,pooled=pooled,paired=pairs,timing=timing,proof=proof,calibration=frozen,
        no_fitting_or_selection=True,ground_truth_at_inference=False,seconds=time.monotonic()-start,limitation=p['limitation']))
    write(run/'status.json',dict(state='completed',updated=now()))

if __name__=='__main__':main()
