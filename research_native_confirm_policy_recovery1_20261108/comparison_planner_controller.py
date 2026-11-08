import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_native_confirm_policy_20261108.common import *


def main():
    import psutil
    root=read(RUN/'started.json')
    while not (RUN/'data_stage_complete.json').exists():
        if datetime.now(MSK)>=datetime(2026,11,8,13,55,tzinfo=MSK):return
        try:p=psutil.Process(root['pid']);assert abs(p.create_time()-root['created'])<.01
        except psutil.NoSuchProcess:raise RuntimeError('Spatial queue ended before completion; preserve and recover failed stage.')
        time.sleep(5)
    sources=read(RUN/'feature_phase.json')['sources'].copy()
    for path in Path(__file__).parent.glob('*.py'):sources[path.relative_to(ROOT).as_posix()]=sha(path)
    for name in ['features_complete.json','data_complete.json','fit_geometry_diagnostic.json','evaluate26_metadata_complete.json','data_stage_complete.json']:
        path=RUN/name;sources[path.relative_to(ROOT).as_posix()]=sha(path)
    for key,h in read(RUN/'features_complete.json')['files'].items():sources[(RUN/'features'/(key+'.npz')).relative_to(ROOT).as_posix()]=h
    videos={**read(TRACE/'native_trace_complete.json')['videos'],**read(RUN/'evaluate26_metadata_complete.json')['videos']}
    for video,r in videos.items():
        sources.update(r['artifacts'])
        for n in [f'data/bdd100k/labels/box_track_20/train/{video}.json',f'runs/train_cache/{video}/alignment.json']:sources[n]=sha(ROOT/n)
    sources['runs/spatial_identity_maps_20261107/image_normalization.json']=sha(ROOT/'runs/spatial_identity_maps_20261107/image_normalization.json')
    write(RUN/'comparison_phase.json',dict(registered=datetime.now(MSK).isoformat(),sources=sources),new=True)
    write(RUN/'comparison_registration.json',dict(registered=datetime.now(MSK).isoformat(),sources=sources,stages=[dict(name='scores',script='research_native_confirm_policy_20261108/score.py'),dict(name='comparison',script='research_native_confirm_policy_recovery1_20261108/compare.py'),dict(name='independent_audit',script='research_native_confirm_policy_20261108/audit.py')]),new=True)
    write(RUN/'comparison_planner_result.json',dict(passed=True),new=True)


if __name__=='__main__':main()
