"""New memory policies on immutable, previously audited GPU FP32 inputs.

No image inference, training, or completed last-policy tracker is repeated.
Native runs only as the new shadow state control; scientific references are reused.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
from types import SimpleNamespace
import sys,time,shutil,faulthandler
from datetime import datetime
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261029 import RUN,service

def compare(a,b):
    import numpy as np
    assert set(a)==set(b) and len(a)==20
    vids=sorted(a);delta=np.array([a[v]['HOTA']-b[v]['HOTA'] for v in vids])
    samples=delta[np.random.default_rng(1).integers(0,20,(20000,20))].mean(1)
    return dict(mean_per_video=float(delta.mean()),ci95=np.quantile(samples,[.025,.975]).tolist(),
        ci_bonferroni_9875=np.quantile(samples,[.00625,.99375]).tolist(),better=int((delta>0).sum()),n=20)

def main():
    audit='--audit' in sys.argv
    import numpy as np,torch
    from tracking.query_memory_write import MemoryWriteTracker
    from tracking.query_appearance_comparison import assert_states_equal
    from tracking.frame_masa_appearance import FrameMasaServer
    from tracking.offline_association import _DetSlice
    from scripts.run_bdd_video_rate_experiment import load_detection_cache,detections_for_frame,scalabel_frame
    from evaluation.hota_eval import evaluate_hota
    gpu=RUN/'memory_write_val';folder=RUN/('memory_write_val_audit' if audit else 'memory_write_val')
    p=service.read(gpu/'protocol.json');old=RUN/'masa_query_fp32_recovery1'
    for f,h in p['sha256'].items():assert service.sha(ROOT/f)==h,f
    if audit:
        c=service.read(gpu/'controller_result.json');assert c['exit_code']==0 and c['all_owned_ended']
        assert not any(service.matching(k,v) for k,v in c['owned'].items())
        assert service.read(gpu/'report.json')['complete']
    else:
        c=service.read(RUN/'memory_write_val_preparation/controller_result.json')
        assert c['exit_code']==0 and c['all_owned_ended']
    torch.set_default_device('cpu');torch.set_num_threads(2)
    deadline=datetime.fromisoformat(service.read(folder/'execution_deadline.json')['deadline']).timestamp()
    def status(**kw):
        assert time.time()<deadline-30,'Global save deadline'
        assert shutil.disk_usage(ROOT).free>5_000_000_000,'Binding disk reserve5GB'
        faulthandler.cancel_dump_traceback_later();faulthandler.dump_traceback_later(600,exit=True)
        service.write(folder/'status.json',dict(updated=service.now(),**kw))
    class Replay(FrameMasaServer):
        def __init__(self):super().__init__(None);self.reads=0
        def features(self,tokens,null_seed=None):
            assert null_seed is None
            for token in tokens:
                assert token.frame<=self.frame
                if token.vector is None:
                    assert token.frame==self.frame
                    token.vector=self.values[tuple(token.crop.tolist())].copy();self.reads+=1
            return np.stack([t.vector for t in tokens])
    names=p['new_variants'];reference_names={'native':'native','r50_last':'r50_1333','detic_last':'detic_1333'}
    per={n:{} for n in [*reference_names,*names]};checked=[];began=time.monotonic()
    for vi,source in enumerate(p['sources']):
        video=source['video'];prior=old/'videos'/video;out=gpu/'videos'/video;out.mkdir(parents=True,exist_ok=True)
        original=service.read(prior/'report.json')
        for n,k in reference_names.items():per[n][video]=original['metrics'][k]
        inputs=dict(protocol_sha256=service.sha(gpu/'protocol.json'),original_video_report_sha256=service.sha(prior/'report.json'))
        marker=folder/'completed'/(video+'.json') if audit else out/'report.json'
        if marker.exists():
            item=service.read(marker);assert item['inputs']==inputs
            assert all(service.sha(ROOT/f)==h for f,h in item['sha256'].items())
            for n in names:per[n][video]=item['metrics'][n]
            checked.append(item);continue
        ready=out/'tracking_completed.json'
        # A completed new tracking pass survives a metrics failure without replay.
        perform_tracking=audit or not ready.exists()
        seconds=time.monotonic()
        if perform_tracking:
            cache=load_detection_cache(ROOT/source['detections']);baseline=service.read(ROOT/source['baseline'])
            align=service.read(ROOT/source['alignment'])['matches'];anchors={int(x['video_frame_index']):i for i,x in enumerate(align)}
            vectors=np.load(prior/'vectors.npz');native_expected=np.load(prior/'raw.npz')['native']
            servers={n:Replay() for n in ['r50','detic']};args=SimpleNamespace(**p['config'])
            trackers={'native_shadow_reference':MemoryWriteTracker(args),
                'zero':MemoryWriteTracker(args,servers['detic'],p['appearance_weight'],True,'cautious',p['beta'])}
            for n in names:
                encoder,policy=n.split('_');trackers[n]=MemoryWriteTracker(args,servers[encoder],p['appearance_weight'],False,policy,p['beta'])
            rows={n:[] for n in trackers};raw={n:[] for n in trackers}
            for frame in range(int(cache['frame_count'])):
                if frame%50==0:status(stage='Independent new-policy CPU replay' if audit else 'New memory policies on exact cached GPU vectors',index=vi,total=20,video=video,frame=frame,total_frames=int(cache['frame_count']))
                active=frame in anchors
                if active:
                    for encoder,s in servers.items():
                        s.start_frame(frame,None);key='r50_1333' if encoder=='r50' else 'detic_1333'
                        boxes=vectors[f'{frame}_{key}_boxes'];values=vectors[f'{frame}_{key}_vectors']
                        s.values={tuple(b.tolist()):v for b,v in zip(boxes,values)}
                boxes,scores,classes=detections_for_frame(cache,frame);d=_DetSlice(boxes,scores,classes)
                outputs={n:t.update_query_frame(d,active) for n,t in trackers.items()}
                assert np.array_equal(outputs['native_shadow_reference'],outputs['zero'])
                assert_states_equal(trackers['native_shadow_reference'],trackers['zero'])
                if active:
                    for s in servers.values():s.finish_frame()
                for n,value in outputs.items():
                    if len(value):raw[n].append(np.column_stack([np.full(len(value),frame),value[:,:7]]).astype(np.float32))
                    if active:
                        i=anchors[frame];rows[n].append(scalabel_frame(video,{k:baseline[i][k] for k in ['name','frameIndex']},value))
            arrays={n:np.concatenate(x) if x else np.zeros((0,8),np.float32) for n,x in raw.items()}
            assert np.array_equal(arrays['native_shadow_reference'],native_expected)
            assert rows['native_shadow_reference']==baseline
            if audit:
                expected=np.load(out/'raw.npz')
                for n,t in trackers.items():
                    assert np.array_equal(arrays[n],expected[n]),(video,n,'raw')
                    assert rows[n]==service.read(out/(n+'.json')),(video,n,'JSON')
                    assert t.events==service.read(out/(n+'_events.json')),(video,n,'cost')
                    assert t.write_events==service.read(out/(n+'_writes.json')),(video,n,'memory')
            else:
                for n,t in trackers.items():
                    service.write(out/(n+'.json'),rows[n]);service.write(out/(n+'_events.json'),t.events);service.write(out/(n+'_writes.json'),t.write_events)
                np.savez_compressed(out/'raw.npz',**arrays)
                files=[f for f in out.iterdir() if f.suffix in ('.json','.npz')]
                service.write(ready,dict(inputs=inputs,sha256={str(f.relative_to(ROOT)):service.sha(f) for f in files},native_zero_raw_state_exact=True,
                    exact_prior_vectors_used=True,encoder_inferences=0,completed_last_policy_reexecutions=0,updated=service.now()))
        saved=service.read(ready);assert saved['inputs']==inputs
        assert all(service.sha(ROOT/f)==h for f,h in saved['sha256'].items())
        metrics={}
        for n in names:
            status(stage='Full per-video metrics',index=vi,video=video,variant=n)
            dest=folder/'metric_inputs'/n;service.write(dest/(video+'.json'),service.read(out/(n+'.json')))
            metrics[n]=evaluate_hota(dest,[video],split='val');per[n][video]=metrics[n]
            if audit:assert metrics[n]==service.read(out/'report.json')['metrics'][n]
        item=dict(video=video,inputs=inputs,metrics=metrics,passed=True,sha256={**saved['sha256'],str(ready.relative_to(ROOT)):service.sha(ready)},
            seconds=time.monotonic()-seconds,raw_json_cost_memory_exact=audit)
        service.write(marker,item);checked.append(item)
        if vi==0 and not audit:
            size=sum(f.stat().st_size for f in out.iterdir() if f.is_file());free=shutil.disk_usage(ROOT).free
            assert 2*size*19+5_000_000_000<free,'Post-first-video disk gate'
            service.write(folder/'price.json',dict(first_complete_video_seconds=item['seconds'],remaining_conservative_seconds=2*item['seconds']*19+300,
                first_saved_bytes=size,required_bytes=2*size*19+5_000_000_000,free_bytes=free,time_gate_advisory_only=True))
    pooled={}
    references=service.read(RUN/'masa_final_checked_record_recovery1/scientific_summary.json')
    for n,k in reference_names.items():pooled[n]=references['pooled'][k]
    for n in names:
        status(stage='Full pooled metrics',variant=n)
        pooled[n]=evaluate_hota(folder/'metric_inputs'/n,[s['video'] for s in p['sources']],split='val')
        if audit:assert pooled[n]==service.read(gpu/'report.json')['pooled'][n]
    comparisons={e+'_'+b:compare(per[e+'_cautious'],per[e+'_'+b]) for e in ['r50','detic'] for b in ['last','mean']}
    native_comparisons={n:compare(per[n],per['native']) for n in names}
    if audit:assert comparisons==service.read(gpu/'report.json')['primary_comparisons']
    assert not torch.cuda.is_initialized()
    result=dict(complete=True,passed=True,videos=20,new_variants=names,reused_references=list(reference_names),checked=checked,
        primary_comparisons=comparisons,native_comparisons=native_comparisons,pooled=pooled,
        main_acceptance=all(x['ci_bonferroni_9875'][0]>0 for x in comparisons.values()),
        encoder_check_reused_source='runs/continuation_20261029/masa_audit_fp32_recovery1/report.json',
        feature_max_difference_reused=references['feature_max_difference'],encoder_inferences=0,
        scientific_scope='Memory write policy in our fixed query tracker, not complete Deep OC-SORT or FlyVis core; old exact GPU FP32 vectors reused, predictor arithmetic unchanged.',
        audit_complete=audit,seconds=time.monotonic()-began,updated=service.now())
    service.write(folder/'report.json',result)
    if audit:service.write(gpu/'independent_audit.json',result)
    faulthandler.cancel_dump_traceback_later()

if __name__=='__main__':main()
