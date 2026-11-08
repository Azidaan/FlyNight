"""Bind the prespecified20-video causal pilot to cached detections and fixed parameters."""
import os,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_identity_assignment_run_20261108.common import *
from research_identity_assignment_run_20261108.full_retina import FULL


def main(obs):
    import shutil
    assert read(RUN/'independent_audit.json')['passed'] and read(RUN/'comparison_report.json')['full_tracker_gate']['proceed']
    parent=read(ROOT/'research_identity_assignment_protocol_20261108/protocol_v2.json');selection=read(RUN/'selection.json')
    old=read(ROOT/'runs/continuation_20261029/cost_use_val/protocol.json')
    names=parent['cohorts']['future_val20']['videos'];assert names==[r['video'] for r in old['sources']]
    dev={q['video'] for q in read(ROOT/'research_identity_assignment_protocol_20261108/queries.json')['rows']};assert not dev&set(names)
    weights=sha(ROOT/'weights/baseline_seed42/best.pt');videos={};cfg=parent['native_config']
    oldcfg=read(ROOT/'runs/b123_20261027/recovery_motion_control/protocol.json')['configs']['xywh']
    for key,value in cfg.items():assert oldcfg[key]==value,(key,oldcfg[key],value)
    for index,r in enumerate(old['sources']):
        video=r['video'];meta_path=Path(r['detections']).with_suffix('.json');meta=read(ROOT/meta_path);identity=meta['identity']
        assert identity['weights_sha256']==weights and identity['imgsz']==960 and identity['conf']==.01 and identity['iou']==.7
        assert Path(identity['video_path']).resolve()==(ROOT/r['video_path']).resolve()
        assert Path(identity['weights_path']).resolve()==(ROOT/'weights/baseline_seed42/best.pt').resolve()
        movie_hash=sha(ROOT/r['video_path']);assert movie_hash==identity['video_sha256']
        assert not (ROOT/f'data/bdd100k/labels/box_track_20/train/{video}.json').exists()
        alignment=read(ROOT/r['alignment'])['matches'];truth_name=f'data/bdd100k/labels/box_track_20/val/{video}.json';truth=read(ROOT/truth_name)
        assert len(alignment)==len(truth)
        clock={str(int(a['video_frame_index'])):dict(name=g['name'],frameIndex=g['frameIndex']) for a,g in zip(alignment,truth)}
        assert len(clock)==len(truth) and max(map(int,clock))<meta['frame_count']
        runtime=dict(video=video,frames=meta['frame_count'],anchors=clock,annotations_excluded=True)
        write(FULL/'runtime_inputs'/(video+'.json'),runtime,new=True)
        fps_path=f'runs/z13_val20_matched_20261024/cache/{video}/motion/flyvis_tiled.json';fps=read(ROOT/fps_path)['fps']
        fields=['detections','alignment','video_path','raw_array','baseline']
        sources={r[k]:sha(ROOT/r[k]) for k in fields}
        sources.update({meta_path.as_posix():sha(ROOT/meta_path),truth_name:sha(ROOT/truth_name),fps_path:sha(ROOT/fps_path)})
        videos[video]=dict(**{k:r[k] for k in fields},frames=meta['frame_count'],fps=fps,sources=sources,baseline_metrics=r['baseline_metrics'])
        status('full_pilot_preflight',complete=index+1,total=len(names),video=video,guard=obs.read())
    assert shutil.disk_usage(ROOT).free>20*1024**3
    code=['research_identity_assignment_run_20261108/'+n for n in ['full_preflight.py','full_retina.py','full_planner.py','full_run.py']]
    write(FULL/'protocol.json',dict(registered=datetime.now(MSK).isoformat(),authorization='Current direct execution request, gated continuation of protocol_v2 after independently verified positive screening. No additional parameter selection.',
        parent_sha256=sha(ROOT/'research_identity_assignment_protocol_20261108/protocol_v2.json'),selection_sha256=sha(RUN/'selection.json'),
        conditional_report_sha256=sha(RUN/'comparison_report.json'),videos=videos,parameters=selection['parameters'],native_config=cfg,
        inference=parent['inference'],deadline='2026-11-09T07:40:00+03:00',historical=True,independent_final_test=False,
        core_checkpoint_sha256=sha(ROOT/'runs/flyvis/results/flow/0000/000/best_chkpt'),detector_checkpoint_sha256=weights,
        source_hashes={n:sha(ROOT/n) for n in code},
        aliases={'G_SHIFT':'G','G_IMAGE':'G','G_DIS':'G'},
        flow_note='All three frozen beta0 arms alias G exactly. No val20 DIS vectors are required or newly calculated. Active spatial arms use the same frozen880-node legacy DIS interpolation hull; no signal-dependent mask.',
        primary_hota='Official detection-weighted HOTA, pooled across videos/classes; paired video resampling recomputes TP/FP/FN and TP-weighted association components, never averages percentages.',
        memory='One external system-reserve guardian for the parent and its registered sequential CPU children; one project CUDA lock. Processed float32 video scratch released only after all relevant causal outputs and retinal views are sealed.'),new=True)
    write(FULL/'preflight.json',dict(passed=True,videos=len(videos),detector_weights_sha256=weights,disjoint_from_calibration_and_evaluate26=True,
        detector_training_split='val video IDs absent from train label files; cached identity exactly matches current weights,960/.01/.7.',
        fixed_parameters_sha256=sha(RUN/'selection.json')),new=True)


if __name__=='__main__':main(guard())
