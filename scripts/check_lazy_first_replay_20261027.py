"""Коротко проверить будущий аудитор на уже готовом первом видео без нового GPU/TrackEval."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
import sys,ast,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.recovery_research_service_20261027 import RUN,read,write,sha,now

def main():
    import numpy as np
    from tracking.lazy_appearance_association import LazyAppearanceXYWH
    from tracking.xywh_bytetrack import XYWHByteTrack
    from tracking.offline_association import _DetSlice
    from scripts.run_bdd_video_rate_experiment import load_detection_cache,detections_for_frame,scalabel_frame
    from types import SimpleNamespace
    path=ROOT/('scripts/audit_lazy_appearance_association_recovery1_20261027.py' if '--recovery1' in sys.argv else 'scripts/audit_lazy_appearance_association_20261027.py');tree=ast.parse(path.read_text(encoding='utf-8'))
    parent=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='audit');cls=next(n for n in parent.body if isinstance(n,ast.ClassDef) and n.name=='Replay')
    scope=dict(np=np,LazyAppearanceXYWH=LazyAppearanceXYWH,XYWHByteTrack=XYWHByteTrack)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),str(path),'exec'),scope);Replay=scope['Replay']
    original_method=Replay.get_dists
    def diagnostic_method(self,tracks,detections):
        try:return original_method(self,tracks,detections)
        except AssertionError:
            e=self.saved_events[self.cursor];d=XYWHByteTrack.get_dists(self,tracks,detections)
            ii=[next(i for i,t in enumerate(tracks) if int(t.track_id)==ident) for ident in e['track_ids']]
            jj=[next(j for j,t in enumerate(detections) if np.array_equal(t.xyxy,b)) for b in e['candidate_boxes']]
            print(dict(frame=self.frame_id-1,event_cursor=self.cursor,dtype=str(d.dtype),ii=ii,jj=jj,original=d[np.ix_(ii,jj)].tolist(),saved=e['original'],
                maximum_difference=float(np.max(abs(d[np.ix_(ii,jj)]-np.array(e['original']))))),flush=True)
            raise
    Replay.get_dists=diagnostic_method
    p=read(RUN/'lazy_appearance_dev/protocol.json');source=p['sources'][0];v=source['video'];base=read(ROOT/source['baseline']);align=read(ROOT/source['alignment'])['matches']
    anchors={int(a['video_frame_index']):i for i,a in enumerate(align)};cache=load_detection_cache(ROOT/source['detections']);checked={};start=time.monotonic()
    for name,weight in [('baseline',0.),('appearance_w0.1',.1),('appearance_w0.3',.3)]:
        expected=read(RUN/'lazy_appearance_dev'/name/(v+'.json'));events=read(RUN/'lazy_appearance_dev/events'/name/(v+'.json'))
        tr=Replay(SimpleNamespace(**p['config']),events,weight);rows=[]
        for frame in range(int(cache['frame_count'])):
            assert time.monotonic()-start<60,'Short replay limit'
            xyxy,conf,classes=detections_for_frame(cache,frame);out=tr.update(_DetSlice(xyxy,conf,classes))
            if frame in anchors:
                ref=anchors[frame];template={k:base[ref][k] for k in ['name','frameIndex']};rows.append(scalabel_frame(v,template,np.asarray(out).reshape(-1,8)))
        assert rows==expected and tr.cursor==len(events),(name,'replay differs');checked[name]=dict(rows=len(rows),events=len(events))
    result=dict(passed=True,first_video=v,checked=checked,seconds=time.monotonic()-start,auditor_source_sha256=sha(path),updated=now())
    write(RUN/'lazy_first_replay.json',result);print(result)

if __name__=='__main__':main()
