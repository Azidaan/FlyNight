"""One shared feature set on immutable native trajectories; reuse completed maps."""
import os,sys,time,shutil
os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from collections import Counter
from research_bbox_normalized_full_20261108.common import *


def main(obs):
    import numpy as np
    from tracking.gpu_lock import GpuLock
    full_phase();p=read(FULL/'protocol.json');assert read(FULL/'native_trace_complete.json')['passed'];manifest={};counts=Counter();first=None
    with GpuLock('flyvis_fovea_gpu'):
        import torch
        torch.cuda.init();assert torch.cuda.is_available();torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
        import research_singleton_free_20261108.full_features as module
        from tracking.flyvis_adapter import activate_flyvis
        from scripts.z13_target_model import build_arm
        from tracking.own_template_verification import correspondence,absolute_score
        from tracking.spatial_identity_assignment import frame_points
        from research_identity_assignment_run_20261108.full_planner import dis_domain
        module.FULL=FULL;module.status=status
        def existing_motion(video,observation):
            entry=p['motion_sources'][video];path=ROOT/entry['path'];assert sha(path)==entry['sha256'];return path
        module.recover_dis=existing_motion
        backbone=None;weights=None;g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json')
        for number,video in enumerate(p['videos']):
            marker=FULL/'feature_receipts'/(video+'.json')
            if marker.exists():
                r=read(marker)
                for n,h in r['artifacts'].items():assert sha(ROOT/n)==h
                manifest[video]=sha(marker);counts.update(r['counts']);continue
            metadata=read(FULL/'native_trace'/(video+'.json'));requests={};events=[]
            for event in metadata['events']:
                for e in event['edges']:
                    if e.get('unavailable'):continue
                    row=e['row_data'];ref=[[h['frame'],h['crop']] for h in row['history']];cur=ref+row['search'];rk=sequence_key(video,ref);ck=sequence_key(video,cur);requests.setdefault(rk,ref);requests.setdefault(ck,cur);events.append((event['frame'],e,rk,ck))
            folder=FULL/'features'/video;folder.mkdir(parents=True,exist_ok=True);reuse={};local=Counter()
            # Priority was fixed before this stage: audited own-template maps,
            # then the earlier native-precision maps, with hashes from receipts.
            for name,entry in p['ready_feature_sources'].get(video,{}).items():
                key=name.split('.')[0]
                if key not in requests:continue
                dest=folder/name;source=ROOT/entry['path'];assert sha(source)==entry['sha256']
                if not dest.exists():shutil.copyfile(source,dest)
                assert sha(dest)==entry['sha256'];reuse[dest.relative_to(ROOT).as_posix()]=entry;local['ready_maps_reused']+=1
            need_core=any(not (folder/(key+'.C.npy')).exists() for key in requests)
            if need_core and backbone is None:
                activate_flyvis(device='cuda');_,_,backbone=build_arm('C',torch,None);assert backbone.dev.type=='cuda' and not backbone.net.training;weights={k:v.detach().clone() for k,v in backbone.net.named_parameters()}
            reps={kind:module.Representations(video,kind,obs,backbone) for kind in ['C','R1','IMAGE','DIS']};renderer=module.Receptors(video,obs)
            for kind in ['C','R1','IMAGE']:reps[kind].renderer=renderer
            values={};lasts={key:seq[-1][0] for key,seq in requests.items()}
            for key,sequence in requests.items():
                for kind,rep in reps.items():values.setdefault(key,{})[kind]=rep.get(sequence,lasts[key])
                local['sequences']+=1
            scores={}
            for frame,e,rk,ck in events:
                row=e['row_data'];cur=values[ck];ref=values[rk];cv=dis_domain(frame_points(g['points'],row['current_crop'],g['eye_size']));rv=dis_domain(frame_points(g['points'],row['history'][-1]['crop'],g['eye_size']));pair=correspondence(e['transform'],row['history'],row['current_crop'],e['box'],g,cv,rv);assert pair is not None and len(pair['unique'])==e['support'];score={}
                for arm in ['SUPPORT','RANDOM','IMAGE','R1','DIS','FLYVIS','PERM']:
                    kind='C' if arm in ['SUPPORT','RANDOM','FLYVIS','PERM'] else arm;score[arm]=absolute_score(arm,cur[kind],ref[kind],pair,e['transform']['scale'],video=video,tid=e['old_tid'],last_frame=row['history'][-1]['frame'],frame=frame,column=e['column'])
                scores[f"{frame}|{e['old_tid']}|{e['new_tid']}"]=dict(scores=score,support=e['support'],reference_key=rk,current_key=ck)
            for event in metadata['events']:
                for e in event['edges']:
                    if e.get('unavailable'):scores[f"{event['frame']}|{e['old_tid']}|{e['new_tid']}"]=dict(scores={a:None for a in ['SUPPORT','RANDOM','IMAGE','R1','DIS','FLYVIS','PERM']},support=0)
            files={n:h for rep in reps.values() for n,h in rep.files.items()};local.update({kind+'_new_maps':rep.new for kind,rep in reps.items()});local['edges']=len(events)
            # Receptors hashes the native-precision file before first use. Also
            # verify it after use; cached maps alone do not need processed IO.
            if renderer._gray is not None:
                processed=module.OLD/'processed'/(video+'.float64');assert sha(processed)==read(processed.with_suffix('.json'))['sha256']
            scorepath=FULL/'scores'/(video+'.json');write(scorepath,dict(passed=True,video=video,edges=scores),new=True);files[scorepath.relative_to(ROOT).as_posix()]=sha(scorepath)
            r=dict(passed=True,video=video,counts=dict(local),artifacts=files,reused=reuse,shared_all_arms=True);write(marker,r,new=True);manifest[video]=sha(marker);counts.update(local)
            if backbone is not None:assert all(torch.equal(value,weights[name]) for name,value in backbone.net.named_parameters())
            status('native_confirm_full_shared_features',video=video,complete=number+1,total=len(p['videos']),counts=dict(counts),guard=obs.read())
    full_phase();write(FULL/'features_complete.json',dict(passed=True,videos=manifest,counts=dict(counts),all_native_histories_shared=True,no_model_training=True),new=True)


if __name__=='__main__':main(guard())
