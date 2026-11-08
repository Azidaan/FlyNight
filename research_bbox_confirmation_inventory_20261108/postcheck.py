"""Close the input integrity chain after the completed technical recovery."""
import os,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_bbox_confirmation_inventory_20261108.common import *


def main(obs):
    assert read(PREVIOUS/'full/queue_result.json')['passed'];folder=PREVIOUS/'full_recovery1';complete=read(folder/'views_complete.json');checked=0
    for video,h in complete['videos'].items():
        path=folder/'views'/(video+'.json');assert sha(path)==h
        for n,value in read(path)['artifacts'].items():assert sha(ROOT/n)==value,n;checked+=1
    for n,r in read(folder/'partial_maps.json')['files'].items():assert sha(ROOT/n)==r['sha256'],n
    allowed={r['path'] for r in read(folder/'reconstructed_binary_check.json')['files']};actual={p.relative_to(ROOT).as_posix() for p in (PREVIOUS/'full/processed').glob('*.float64')};assert actual==allowed,'Unexpected full decoded-video fallback.'
    write(RUN/'completed_previous_input_audit.json',dict(passed=True,checked_artifacts=checked,all_prepared_views_and_prior_maps_unchanged=True,no_new_full_decoded_video_fallback=True,source_sha256=sha(folder/'views_complete.json')),new=True)


if __name__=='__main__':main(guard())
