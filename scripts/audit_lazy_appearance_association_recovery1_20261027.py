"""Воспроизвести все назначения и состояния по сохранённым стоимостям без GPU."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
from types import SimpleNamespace
import sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.recovery_research_service_20261027 import RUN as BASE,read,write,sha,matching,now

def audit(folder):
    import numpy as np
    from tracking.lazy_appearance_association import LazyAppearanceXYWH
    from tracking.offline_association import _DetSlice
    from scripts.run_bdd_video_rate_experiment import load_detection_cache,detections_for_frame,scalabel_frame
    from tracking.xywh_bytetrack import XYWHByteTrack
    from evaluation.hota_eval import evaluate_hota
    run=BASE/folder;p=read(run/'protocol.json');r=read(run/'report.json');controller=read(run/'controller_result.json')
    assert r['complete'] and controller['exit_code']==0 and controller['all_owned_ended'] and not controller['timed_out']
    assert not any(matching(k,v) for k,v in controller['owned'].items())
    for name,h in p['sha256'].items():assert sha(ROOT/name)==h,name
    dev=p['phase']=='dev';frozen=read(BASE/'lazy_appearance_frozen.json');weight=frozen['weight'];checked=0;checked_events=0;began=time.monotonic()
    if dev:
        best=max(r['mean_per_video'],key=lambda n:(r['mean_per_video'][n],-(0 if n=='baseline' else float(n.rsplit('_w',1)[1]))));assert frozen['selected']==best and r['chosen']==frozen
    else:assert r['frozen']==frozen and r['batchnorm_buffers_unchanged'] and not r['ground_truth_used_by_predictor']
    class Replay(LazyAppearanceXYWH):
        def __init__(self,args,events,weight):
            super().__init__(args,weight=0);self.saved_events=events;self.cursor=0;self.expected_weight=weight
        def get_dists(self,tracks,detections):
            original=XYWHByteTrack.get_dists(self,tracks,detections)
            if not self._inside_first or self.cursor==len(self.saved_events):return original
            event=self.saved_events[self.cursor]
            if event['frame']!=self.frame_id-1:return original
            ii=[next(i for i,t in enumerate(tracks) if int(t.track_id)==ident) for ident in event['track_ids']]
            jj=[]
            # Рамка сама по себе не уникальна: разные YOLO-детекции могут иметь
            # одинаковые координаты, но разные баллы и геометрические стоимости.
            for col,b in enumerate(event['candidate_boxes']):
                options=[j for j,d in enumerate(detections) if j not in jj and np.array_equal(np.asarray(d.xyxy),b)
                         and np.allclose(original[ii,j],np.asarray(event['original'])[:,col],atol=1e-7,rtol=0)]
                assert options,(event['frame'],col,'No unique original candidate')
                jj.append(options[0])
            old=original[np.ix_(ii,jj)];assert np.allclose(old,event['original'],atol=1e-7,rtol=0)
            eligible=np.array(event['eligible'],bool);assert np.array_equal(eligible,old<=self.args.match_thresh)
            assert all(f<event['frame'] for f in event['past_frames'])
            appearance=np.array(event['appearance']);assert np.isfinite(appearance).all() and ((appearance>=0)&(appearance<=1)).all()
            expected=np.where(eligible,old+self.expected_weight*appearance,old)
            # Float32 calculation in production; saved values must agree at that precision.
            assert np.allclose(expected,event['updated'],atol=1e-7,rtol=0)
            out=original.copy();out[np.ix_(ii,jj)]=np.asarray(event['updated'],np.float32);self.cursor+=1
            return out
    for source in p['sources']:
        v=source['video'];base=read(ROOT/source['baseline']);assert read(run/'baseline'/(v+'.json'))==base
        align=read(ROOT/source['alignment'])['matches'];anchors={int(a['video_frame_index']):i for i,a in enumerate(align)};cache=load_detection_cache(ROOT/source['detections'])
        for name in r['per_video']:
            events=read(run/'events'/name/(v+'.json'));w=(0 if name=='baseline' else float(name.rsplit('_w',1)[1])) if dev else (0 if name=='baseline' else weight)
            tracker=Replay(SimpleNamespace(**p['config']),events,w);actual=read(run/name/(v+'.json'));rows=[]
            for frame in range(int(cache['frame_count'])):
                xyxy,conf,classes=detections_for_frame(cache,frame);out=tracker.update(_DetSlice(xyxy,conf,classes))
                if frame in anchors:
                    ref=anchors[frame];template={k:base[ref][k] for k in ['name','frameIndex']};rows.append(scalabel_frame(v,template,np.asarray(out).reshape(-1,8)))
            assert tracker.cursor==len(events) and rows==actual,(v,name,'assignment replay differs');checked+=len(rows);checked_events+=len(events)
        write(BASE/(folder+'_audit')/'status.json',dict(state='running',updated=now(),video=v,checked_output_frames=checked),required=False)
    independent={};decisions={}
    if not dev:
        rng=np.random.default_rng(281030)
        for group,contrasts in r['paired'].items():
            vs=[s['video'] for s in p['sources']] if group=='all40' else p['groups'][group];independent[group]={}
            for key,value in contrasts.items():
                b='baseline' if key=='appearance_vs_baseline' else key.removeprefix('appearance_vs_')
                d=np.array([r['per_video']['appearance'][v]['HOTA']-r['per_video'][b][v]['HOTA'] for v in vs]);assert abs(d.mean()-value['mean_per_video'])<1e-10 and value['n']==len(vs)
                boot=d[rng.integers(0,len(vs),(5000,len(vs)))].mean(1);independent[group][key]=dict(mean_per_video=float(d.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),n=len(vs))
            decisions[group+'_positive_vs_baseline_and_all3_nulls']=weight>0 and all(x['ci95'][0]>0 for x in contrasts.values())
    result=dict(passed=True,phase=p['phase'],complete_videos=len(p['sources']),all_assignment_outputs_replayed=True,checked_output_frames=checked,checked_events=checked_events,
        no_future_history_in_events=True,source_hashes_verified=True,independent_pairs=independent,prespecified_decision=decisions,report_sha256=sha(run/'report.json'),seconds=time.monotonic()-began,updated=now())
    write(run/'independent_audit.json',result);write(BASE/(folder+'_audit')/'report.json',result);return result
