"""Recover only missing required views; verify cached views against native-dtype pixels."""
import os,sys,shutil
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from collections import defaultdict,Counter
from research_native_confirm_full_recovery1_20261108.common import *


def point_key(frame,crop):return hashlib.sha256(json.dumps([int(frame),list(map(int,crop))],separators=(',',':')).encode()).hexdigest()[:32]


def main(obs):
    import numpy as np,torch,cv2
    import research_singleton_free_20261108.full_features as module
    from tracking.retina import make_retina
    torch.set_num_threads(2);module.FULL=FULL;module.status=status;recovery_phase();full_phase();p=read(FULL/'protocol.json');norm=read(ROOT/'runs/spatial_identity_maps_20261107/image_normalization.json');width={'C':8,'R1':34,'IMAGE':len(norm['mean']),'DIS':2};manifest={};totals=Counter()
    # The first attempt reconstructed a released temporary binary correctly.
    # Compare it to the original immutable checksum, not a retroactive new one.
    restored=[]
    for path in (FULL/'processed').glob('*.float64'):
        video=path.stem;old=BASE/'full_precision/processed'/(video+'.json');expected=read(old)['sha256'];actual=sha(path);assert actual==expected and actual==read(path.with_suffix('.json'))['sha256'];restored.append(dict(video=video,path=path.relative_to(ROOT).as_posix(),sha256=actual,equals_original_recorded_hash=True))
    write(RECOVERY/'reconstructed_binary_check.json',dict(passed=True,files=restored),new=True)
    for number,(video,entry) in enumerate(p['videos'].items()):
        marker=RECOVERY/'views'/(video+'.json')
        if marker.exists():
            r=read(marker)
            for n,h in r['artifacts'].items():assert sha(ROOT/n)==h
            totals.update(r['counts']);manifest[video]=sha(marker);continue
        metadata=read(FULL/'native_trace'/(video+'.json'));requests={}
        for event in metadata['events']:
            for e in event['edges']:
                row=e['row_data'];ref=[[h['frame'],h['crop']] for h in row['history']]
                for seq in [ref,ref+row['search']]:requests.setdefault(sequence_key(video,seq),seq)
        folder=FULL/'features'/video;folder.mkdir(parents=True,exist_ok=True);counts=Counter();artifacts={};required=set()
        for name,source in p['ready_feature_sources'].get(video,{}).items():
            if name.split('.')[0] not in requests:continue
            src=ROOT/source['path'];assert sha(src)==source['sha256'];dest=folder/name
            if not dest.exists():shutil.copyfile(src,dest)
            assert sha(dest)==source['sha256'];counts['ready_complete_maps_reused']+=1
        for key,seq in requests.items():
            for kind in ['C','R1','IMAGE','DIS']:
                path=folder/(key+'.'+kind+'.npy')
                if path.exists():
                    value=np.load(path,allow_pickle=False);assert value.shape==(721,width[kind]) and np.isfinite(value).all();artifacts[path.relative_to(ROOT).as_posix()]=sha(path)
                elif kind in ['C','R1']:required.update((int(f),tuple(map(int,c))) for f,c in seq)
                elif kind=='IMAGE':required.add((int(seq[-1][0]),tuple(map(int,seq[-1][1]))))
        destination=FULL/'retinal'/video;destination.mkdir(parents=True,exist_ok=True);known={};origins={}
        for frame,crop in sorted(required):
            key=point_key(frame,crop);dest=destination/(key+'.npy')
            if not dest.exists():
                for old in [PARENT/'retinal'/video/dest.name,BASE/'full_precision/retinal'/video/dest.name]:
                    if old.exists():
                        before=sha(old);value=np.load(old,allow_pickle=False);assert value.shape==(721,) and value.dtype==np.float16 and np.isfinite(value).all();shutil.copyfile(old,dest);assert sha(dest)==before;origins[dest.relative_to(ROOT).as_posix()]=dict(source=old.relative_to(ROOT).as_posix(),sha256_read_now=before,not_a_retroactive_old_run_checksum=True);break
            if dest.exists():
                value=np.load(dest,allow_pickle=False);assert value.shape==(721,) and value.dtype==np.float16 and np.isfinite(value).all();known[(frame,crop)]=value;counts['ready_views_reused']+=1
        if required:
            source=ROOT/entry['video_path'];assert sha(source)==entry['sources'][entry['video_path']];cap=cv2.VideoCapture(str(source));assert cap.isOpened();retina=make_retina('lamina',entry['fps'],blur_downscale=4);renderer=module.Receptors(video,obs);byframe=defaultdict(list)
            for frame,crop in sorted(required):byframe[frame].append(crop)
            for frame in range(max(byframe)+1):
                ok,image=cap.read();assert ok and image.shape[:2]==(720,1280);processed=retina.process(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY));assert str(processed.dtype)==('float32' if frame==0 else 'float64')
                for crop in byframe.get(frame,[]):
                    value=renderer.render(processed,crop);dest=destination/(point_key(frame,crop)+'.npy')
                    if (frame,crop) in known:assert np.array_equal(value,known[(frame,crop)]),(video,frame,crop,'Ready-view numerical parity failed');counts['ready_views_numerically_verified']+=1
                    else:
                        with dest.open('xb') as f:np.save(f,value,allow_pickle=False)
                        counts['new_missing_views']+=1
                    artifacts[dest.relative_to(ROOT).as_posix()]=sha(dest)
                if frame%200==0:status('recover_required_native_precision_views',video=video,frame=frame,total=max(byframe)+1,counts=dict(counts),guard=obs.read())
            cap.release();assert sha(source)==entry['sources'][entry['video_path']]
        r=dict(passed=True,video=video,artifacts=artifacts,counts=dict(counts),cached_view_origins=origins,new_full_video_binary_created=False);write(marker,r,new=True);manifest[video]=sha(marker);totals.update(counts);status('required_views_video_complete',video=video,complete=number+1,total=len(p['videos']),counts=dict(totals),guard=obs.read())
    assert not torch.cuda.is_initialized();recovery_phase();full_phase();write(RECOVERY/'views_complete.json',dict(passed=True,videos=manifest,counts=dict(totals),scope='Missing required views recovered directly from movies with native frame dtype. Existing needed views verified bitwise. Existing complete neural maps reused; no network, detector or tracker run. No new decoded full-video binaries.'),new=True)


if __name__=='__main__':main(guard())
