"""Evaluator-only sparse labels. No interpolation and no annotation-driven references."""
from collections import Counter
import numpy as np
from research_identity_assignment_run_20261108.common import *
from scripts.association_log import label
from datasets.bdd100k_loader import CLASS_NAMES


class Labels:
    def __init__(self,video,export):
        self.video=video;self.sources={};self.export=export;self.current={};self.rivals={}
        def load(name):self.sources[name]=sha(ROOT/name);return read(ROOT/name)
        annotations=load(f'data/bdd100k/labels/box_track_20/train/{video}.json')
        aligned=load(f'runs/train_cache/{video}/alignment.json')['matches'];assert len(annotations)==len(aligned)
        self.truth={}
        for fr,a in zip(annotations,aligned):
            objects=[]
            for l in fr.get('labels') or []:
                if l['category'] not in CLASS_NAMES.values() or (l.get('attributes') or {}).get('crowd',False):continue
                b=l['box2d'];objects.append((str(l['id']),l['category'],np.array([b['x1'],b['y1'],b['x2'],b['y2']])))
            self.truth[int(a['video_frame_index'])]=objects
        self.supervision={r['uid']:r for r in load(f'runs/z13_temporal_pair_readiness_20261025/supervision/{video}.json')['rows']}
        raw=ROOT/f'runs/train_cache/{video}/detections/raw_video.npz';self.sources[raw.relative_to(ROOT).as_posix()]=sha(raw)
        with np.load(raw,allow_pickle=False) as z:self.raw={n:z[n] for n in z.files}

    def current_labels(self,frame):
        frame=int(frame)
        if frame in self.current:return self.current[frame]
        ids=np.flatnonzero(self.raw['frame_idx']==frame);s=self.raw['conf'];out={}
        # High and low waves have distinct native candidate populations.
        for mask in [s[ids]>=.25,(s[ids]>.1)&(s[ids]<.25)]:
            ix=ids[mask];values=label(self.raw['xyxy'][ix],[CLASS_NAMES[int(c)] for c in self.raw['cls'][ix]],self.truth.get(frame,[]))
            out.update({int(i):v for i,v in zip(ix,values)})
        snap=self.export['snapshots'][str(frame)]
        for q in snap['queries']:
            expected=self.supervision[q['uid']]['current_identity']
            assert [out[snap['raw_ids'][i]] for i in q['candidate_high_ids']]==expected,(self.video,frame,q['uid'],'label parity')
        self.current[frame]=out;return out

    def query(self,q,result):
        target=self.supervision[q['uid']]['target_identity'];raw=result['final'].get(str(q['tid']))
        if raw is None:return dict(kind='none',J=0,raw_id=None,identity=None,target=target)
        identity=self.current_labels(q['frame']).get(int(raw))
        if identity is None or target is None:return dict(kind='unknown',J=0,raw_id=raw,identity=identity,target=target)
        correct=identity==target
        return dict(kind='correct' if correct else 'wrong',J=1 if correct else -1,raw_id=raw,identity=identity,target=target)

    def rival_identities(self,frame):
        frame=str(frame)
        if frame in self.rivals:return self.rivals[frame]
        result={}
        for row in self.export['snapshots'][frame]['rows']:
            identities=[]
            for h in row['history']:
                if h['frame'] not in self.truth:continue
                value=label([h['box']],[CLASS_NAMES[row['cls']]],self.truth[h['frame']])[0]
                if value is not None:identities.append(value)
            unique=set(identities);result[str(row['tid'])]=next(iter(unique)) if len(unique)==1 else None
        self.rivals[frame]=result;return result

    def harm(self,frame,baseline,result,query_tids):
        ids=self.current_labels(frame);references=self.rival_identities(frame);c=Counter()
        changed=set(baseline['final'])|set(result['final'])
        for tid in changed:
            before=baseline['final'].get(tid);after=result['final'].get(tid)
            if before==after or int(tid) in query_tids:continue
            c['changed_other_track_assignments']+=1;target=references.get(tid)
            old_id=ids.get(before);new_id=ids.get(after)
            if target is None:c['other_track_reference_unknown']+=1;continue
            if before is not None and old_id is None:c['other_track_previous_assignment_unknown']+=1;continue
            if after is not None and new_id is None:c['other_track_new_assignment_unknown']+=1;continue
            oldJ=0 if before is None else (1 if old_id==target else -1)
            newJ=0 if after is None else (1 if new_id==target else -1)
            c['other_track_delta_J']+=newJ-oldJ
            if newJ>oldJ:c['other_track_improved']+=1
            if newJ<oldJ:c['other_track_worsened']+=1
        return dict(c)
