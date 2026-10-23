"""Service-only FPS metadata overlay; no motion arrays are generated or read."""
from pathlib import Path
import sys, math
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_matched_trackers_20261024 import RUN,VIDEOS,read,write,sha,now
OVERLAY=RUN/'fps_metadata_recovery'


def prepare():
    import cv2
    assert not (OVERLAY/'manifest.json').exists()
    rows={}
    for video in VIDEOS:
        source=ROOT/f'runs/video_rate/{video}/summary.json'
        data=read(source);fps=float(data['protocol']['source_fps'])
        clip=ROOT/f'data/bdd100k/videos/val/{video}.mov'
        cap=cv2.VideoCapture(str(clip));actual=cap.get(cv2.CAP_PROP_FPS);frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));cap.release()
        assert math.isfinite(fps) and fps>0 and fps==actual
        assert frames==data['protocol']['raw_frames']==read(RUN/'protocol.json')['frames'][video]
        assert sha(clip)==data['protocol']['video_sha256']
        old=ROOT/f'runs/video_rate/{video}/motion/flyvis_tiled.json'
        if old.exists():assert float(read(old)['fps'])==fps
        target=OVERLAY/video/'motion/flyvis_tiled.json';write(target,dict(fps=fps))
        rows[video]=dict(fps=fps,frames=frames,summary_sha256=sha(source),video_sha256=sha(clip),previous_metadata_exists=old.exists(),previous_metadata_sha256=sha(old) if old.exists() else None,overlay_sha256=sha(target))
    write(OVERLAY/'manifest.json',dict(at=now().isoformat(),service_sha256=sha(Path(__file__)),rows=rows,change='Only args.flow_root metadata path changes. build_video reads only fps from this path. FPS equals decoded container, immutable video summary and every existing old metadata value exactly. No flow array or estimator settings change.'))


def install():
    manifest=read(OVERLAY/'manifest.json');assert manifest['service_sha256']==sha(Path(__file__))
    for video,row in manifest['rows'].items():
        assert sha(OVERLAY/video/'motion/flyvis_tiled.json')==row['overlay_sha256']
        assert sha(ROOT/f'runs/video_rate/{video}/summary.json')==row['summary_sha256']
    from scripts import build_fly_signatures as builder
    original=builder.parse_args
    def parse_args():
        args=original();assert args.flow_root.resolve()==(ROOT/'runs/video_rate').resolve()
        args.flow_root=OVERLAY;return args
    builder.parse_args=parse_args


if __name__=='__main__':prepare()
