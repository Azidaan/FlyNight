"""ПолныйCPUповтор40видео/32трекеров с проверкой головы и точных назначений."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace
import sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261028 import RUN,service
read,write,sha,now=service.read,service.write,service.sha,service.now


def main():
    import numpy as np,torch,cv2
    from tracking.history_pair_runtime import HistoryAssociationTracker
    from tracking.history_pair_model_deterministic import DeterministicHistoryPairHead
    from tracking.causal_motion_history import CausalMotionHistory
    from scripts.run_bdd_video_rate_experiment import load_detection_cache,detections_for_frame,scalabel_frame
    from evaluation.hota_eval import evaluate_hota
    from scripts.summarize_history_tracker_20261028 import summarize
    source=RUN/'history_tracker_pilot';folder=RUN/'history_tracker_audit';p=read(source/'protocol.json');frozen=read(source/'frozen.json')
    c=read(source/'controller_result.json');assert c['exit_code']==0 and c['all_owned_ended'] and not any(service.matching(k,v) for k,v in c['owned'].items())
    report=read(source/'report.json')
    for file,digest in p['sha256'].items():assert sha(ROOT/file)==digest,file
    if not report['complete']:
        write(folder/'report.json',dict(passed=False,full_series=False,scientific_verdict=None,reason='Full tracker stopped on cost; no complete scientific verdict',updated=now()));return
    assert len(report['completed'])==40 and all(len(v)==40 for v in report['per_video'].values())
    torch.set_num_threads(2);cv2.setNumThreads(1);models={}
    for name,spec in frozen['models'].items():
        assert sha(ROOT/spec['checkpoint'])==spec['sha256'];m=DeterministicHistoryPairHead()
        m.load_state_dict(torch.load(ROOT/spec['checkpoint'],map_location='cpu',weights_only=False)['state']);models[name]=m.eval()
    end=datetime.fromisoformat(read(folder/'execution_deadline.json')['deadline']).timestamp();completed=[];global_max_diff=0.;all_events=0;pixel_rois=0

    class Replay(HistoryAssociationTracker):
        def get_dists(self,tracks,detections):
            start=len(self.events);matrix=super().get_dists(tracks,detections);fresh=self.events[start:]
            expected=self.source_events.get(self.raw_frame,[])
            if not self.first:return matrix
            assert len(fresh)==len(expected),(self.variant,self.raw_frame,'event count differs')
            for new,old in zip(fresh,expected):
                assert all(new[k]==old[k] for k in ['frame','track','det','original','history_frames'])
                diff=abs(new['score']-old['score']);self.max_diff=max(self.max_diff,diff)
                assert diff<=p['head_cpu_gpu_tolerance'],(self.variant,self.raw_frame,diff)
                # Точность головы проверена независимо наCPU. Для побитного
                # повтора дискретных назначений используем записанныйGPUscore.
                i=next(i for i,t in enumerate(tracks) if int(t.track_id)==old['track'])
                j=next(j for j,d in enumerate(detections) if int(d.idx)==old['det'])
                expected_cost=matrix.dtype.type(old['original']+self.alpha*(.5-1/(1+np.exp(-np.clip(old['score'],-80,80))))) if self.alpha else matrix.dtype.type(old['original'])
                assert float(expected_cost)==old['corrected']
                matrix[i,j]=expected_cost;self.checked+=1
            return matrix

    for vi,spec in enumerate(p['sources']):
        started=time.monotonic();v=spec['video'];entry=report['completed'][vi];assert entry['video']==v
        assert all(sha(ROOT/f)==h for f,h in entry['sha256'].items())
        marker=folder/'completed'/(v+'.json')
        key=sha(source/'videos'/v/'report.json')
        if marker.is_file():
            old=read(marker);assert old['input_sha256']==key
            completed.append(old);global_max_diff=max(global_max_diff,old['max_head_difference']);all_events+=old['events'];pixel_rois+=old['pixel_rois'];continue
        visual=source/'videos'/v/'visual'
        embeddings=np.load(visual/'embeddings.npy',mmap_mode='r');images=np.load(visual/'images.npy',mmap_mode='r')
        with np.load(visual/'index.npz') as z:index={k:z[k] for k in z.files}
        assert embeddings.shape==(entry['roi_count'],256) and images.shape==(entry['roi_count'],3,p['image_size'],p['image_size']) and images.dtype==np.uint8
        assert np.isfinite(embeddings).all() and np.allclose(np.linalg.norm(embeddings,axis=1),1,atol=1e-5)
        cache=load_detection_cache(ROOT/spec['detections']);expected=read(ROOT/spec['baseline']);align=read(ROOT/spec['alignment'])['matches'];anchors={int(a['video_frame_index']):i for i,a in enumerate(align)}
        assert np.array_equal(index['query_frame'],list(anchors))
        at={}
        for i,(f,d) in enumerate(zip(index['roi_frame'],index['roi_det'])):at.setdefault(int(f),[]).append((i,int(d)))
        query_at={int(f):i for i,f in enumerate(index['query_frame'])}
        dummy=models['v134_seed42'];args=SimpleNamespace(**p['tracker_config'])
        trackers={'native':Replay(args,dummy,'0',0),'zero':Replay(args,dummy,'134',0)}
        trackers.update({n:Replay(args,m,frozen['models'][n]['variant'],p['alpha']) for n,m in models.items()})
        raw={n:[] for n in trackers};rows={n:[] for n in trackers}
        for name,t in trackers.items():
            values=read(source/'videos'/v/'events'/(name+'.json'));t.source_events={};t.max_diff=0.;t.checked=0
            for event in values:t.source_events.setdefault(event['frame'],[]).append(event)
        check_pixels=vi in [0,20,39];checked_pixels=0
        if check_pixels:
            capture=cv2.VideoCapture(str(ROOT/spec['video_path']));assert capture.isOpened();motion=CausalMotionHistory(**p['motion'])
        for frame in range(int(cache['frame_count'])):
            if time.time()>end-20:raise TimeoutError('History tracker audit original deadline')
            boxes,scores,classes=detections_for_frame(cache,frame);context=None
            if check_pixels:ok,image=capture.read();assert ok
            if frame in anchors:
                q=query_at[frame];items=at.get(frame,[])
                assert [d for _,d in items]==np.flatnonzero(scores>=p['tracker_config']['track_low_thresh']).tolist()
                rois={d:dict(embedding=embeddings[i],image=images[i]) for i,d in items}
                context=dict(rois=rois,camera=index['query_camera'][q],epoch=int(index['query_epoch'][q]),video_size=index['video_size'])
                if check_pixels:
                    state=motion.update(image,frame)
                    assert np.array_equal(np.asarray(state['camera'],np.float32),index['query_camera'][q]) and state['epoch']==index['query_epoch'][q]
                    for i,d in items:assert np.array_equal(motion.crop(boxes[d],image.shape,p['image_size'],p['context']),images[i]);checked_pixels+=1
            for name,t in trackers.items():
                output=t.update_frame(frame,boxes,scores,classes,None if name=='native' else context)
                if name=='native':native_output=output.copy()
                elif name=='zero':assert np.array_equal(output,native_output)
                if len(output):raw[name].append(np.column_stack([np.full(len(output),frame),output[:,:7]]).astype(np.float32))
                if frame in anchors:
                    ref=anchors[frame];rows[name].append(scalabel_frame(v,{k:expected[ref][k] for k in ['name','frameIndex']},output))
                t.events.clear()
            for attr in ['tracked_stracks','lost_stracks','removed_stracks']:
                a=getattr(trackers['native'],attr);b=getattr(trackers['zero'],attr)
                assert len(a)==len(b) and all(x.track_id==y.track_id and np.array_equal(x.mean,y.mean) and np.array_equal(x.covariance,y.covariance) for x,y in zip(a,b))
            if frame%50==0:write(folder/'status.json',dict(state='replay',video=v,index=vi,total=40,frame=frame,events=sum(t.checked for t in trackers.values())+all_events,updated=now()),required=False)
        if check_pixels:capture.release()
        for name in trackers:
            actual=np.concatenate(raw[name]) if raw[name] else np.zeros((0,8),np.float32)
            with np.load(source/'videos'/v/(name+'.npz')) as z:assert np.array_equal(actual,z['outputs'])
            assert rows[name]==read(source/name/(v+'.json'))
            if name=='native':assert rows[name]==expected
            metric=evaluate_hota(source/name,[v],split='val') if name!='zero' else report['per_video']['native'][v]
            assert metric==report['per_video'][name][v],(v,name,'metrics differ')
        item=dict(video=v,input_sha256=key,passed=True,events=sum(t.checked for t in trackers.values()),max_head_difference=max(t.max_diff for t in trackers.values()),pixel_rois=checked_pixels,seconds=time.monotonic()-started)
        write(marker,item);completed.append(item);all_events+=item['events'];global_max_diff=max(global_max_diff,item['max_head_difference']);pixel_rois+=checked_pixels
        if vi==0:
            need=2*item['seconds']*39+300;available=end-time.time();price=dict(first_full_video_seconds=item['seconds'],conservative_remaining_seconds=need,available_seconds=available,proceed=need<available);write(folder/'price.json',price)
            if not price['proceed']:
                write(folder/'report.json',dict(passed=False,full_series=False,scientific_verdict=None,completed=completed,stopped_on_cost=True,updated=now()));return
    pooled={}
    for name in report['pooled']:
        pooled[name]=pooled['native'] if name=='zero' else evaluate_hota(source/name,[s['video'] for s in p['sources']],split='val')
        assert pooled[name]==report['pooled'][name]
    summary=summarize(report['per_video'],pooled,p)
    assert all(summary[k]==report[k] for k in summary)
    assert not torch.cuda.is_initialized()
    write(folder/'report.json',dict(passed=True,full_series=True,videos=40,trackers=32,completed=completed,events=all_events,max_cpu_gpu_head_difference=global_max_diff,pixel_rois=pixel_rois,
        head_cpu_recomputed=True,discrete_replay_uses_validated_recorded_gpu_scores=True,raw_json_assignments_exact=True,native_zero_states_exact=True,all_metrics_paired_repeated=True,main_acceptance=report['main_acceptance'],updated=now()))
    write(source/'independent_audit.json',read(folder/'report.json'))


if __name__=='__main__':main()
