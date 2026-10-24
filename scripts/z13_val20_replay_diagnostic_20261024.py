"""One bounded old-runner replay, retaining the failed strict control unchanged."""
from pathlib import Path
from datetime import datetime
import sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
OLD=ROOT/'runs/z13_val20_matched_20261024/common';RUN=OLD.parent/'common_replay_diagnostic';T0=time.monotonic()


def comparison(a,b):
    import numpy as np
    with np.load(a) as x,np.load(b) as y:
        p=x['outputs'];q=y['outputs'];assert p.shape==q.shape
        different=np.flatnonzero(np.any(p!=q,axis=1));anchor=read(ROOT/'runs/video_rate/b1c66a42-6f7d68ca/alignment.json')['matches'];indices={r['video_frame_index'] for r in anchor}
        return dict(exact=np.array_equal(p,q),shape=list(p.shape),different_rows=len(different),different_frames=sorted(set(p[different,0].tolist())),max_abs=float(np.abs(p-q).max()),identity_class_score_equal=bool(np.array_equal(p[:,[0,5,6,7]],q[:,[0,5,6,7]])),annotation_frames_affected=sorted(set(p[different,0].tolist())&indices),coordinate_deltas=(p[different]-q[different]).tolist())


def prepare():
    from scripts.z13_val20_common_20261024 import ended
    from scripts.z13_localization_labels_20261023 import verify_sources
    import numpy as np
    assert not RUN.exists();e=read(OLD/'controller_result.json');assert e['exit_code']==1 and e['all_owned_ended'] and not e['remaining'];p=read(OLD/'protocol.json');verify_sources(p)
    from scripts.z13_hota_deadline_guard import WindowsProcesses
    ps=WindowsProcesses();c=read(OLD/'controller.json');assert ps.creation(c['pid'])!=c['birth'] and all(ps.creation(int(pid))!=b for pid,b in e['owned'].items())
    v=p['probe_video'];a=OLD/'probe/legacyD/signatures'/v/'signatures.npz';b=ROOT/'runs/z13_matched_trackers_20261024/legacyD/signatures'/v/'signatures.npz'
    write(RUN/'previous_difference.json',comparison(a,b));p={**p,'created':now().isoformat(),'budget_seconds':600,'save_at':c['save_at'],'hard_stop':c['hard_stop'],'stop_request_path':(RUN/'stop.request').relative_to(ROOT).as_posix(),'sha256':{**p['sha256'],'scripts/z13_val20_replay_diagnostic_20261024.py':sha(Path(__file__)),a.relative_to(ROOT).as_posix():sha(a)},'purpose':'One replay with the original matched7 producer (Compiled class) instead of new generic Fixed class; no altered arithmetic,precision,tolerance,seed,data or weights. Compare original vs both fresh outputs. If exact guard still fails, do not accept/proceed to20 merely because numerical difference is small.'};write(RUN/'protocol.json',p)


def worker():
    from scripts import z13_matched_pipeline_20261024 as P
    from scripts.z13_localization_labels_20261023 import verify_sources
    p=read(RUN/'protocol.json');verify_sources(p);P.RUN=RUN;P.STAGE=RUN;P.T0=time.monotonic();v=p['probe_video']
    def status(stage,**kw):
        assert time.monotonic()-T0<600 and now()<datetime.fromisoformat(p['save_at']) and not (RUN/'stop.request').exists();write(RUN/'status.json',dict(state='running',stage=stage,updated=now().isoformat(),elapsed_seconds=time.monotonic()-T0,jobs_done=0,jobs_total=1,**kw))
    P.status=status;status('Единственный повтор прежнего служебного производителя',variant='legacyD '+v)
    P.build_signature('legacyD',v);new=RUN/'legacyD/signatures'/v/'signatures.npz';old=ROOT/'runs/z13_matched_trackers_20261024/legacyD/signatures'/v/'signatures.npz';failed=OLD/'probe/legacyD/signatures'/v/'signatures.npz'
    write(RUN/'report.json',dict(at=now().isoformat(),old_runner_vs_original=comparison(new,old),old_runner_vs_failed=comparison(new,failed),output_sha256=sha(new),no_threshold_changed=True,not_a_hota_or_scientific_effect=True));write(RUN/'status.json',dict(state='completed',stage='Контроль исходного производителя закончен;проверка не ослаблена',updated=now().isoformat(),jobs_done=1,jobs_total=1))


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--worker' in sys.argv:
        try:worker()
        except Exception as e:write(RUN/'status.json',dict(state='failed',stage=repr(e),updated=now().isoformat()));raise
    else:raise RuntimeError('Explicit diagnostic mode required')
