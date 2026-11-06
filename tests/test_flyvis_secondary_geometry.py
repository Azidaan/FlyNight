"""Geometry follows every secondary entry point without reinterpreting old weights."""
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tracking.flyvis_geometry_protocol import (
    add_geometry_arguments, ensure_geometry_output, require_arm_geometry, require_reference_geometry,
)
from tracking.flyvis_motion import GEOMETRY_VERSION, GEOMETRY_VERSIONS


@pytest.mark.parametrize('file',[False,True])
@pytest.mark.parametrize('version',['pixel_centers_v2','resize_v2'])
def test_existing_unversioned_results_reject_new_geometry_without_writing(tmp_path,file,version):
    path=tmp_path/'old.json' if file else tmp_path/'old'
    if file:path.write_text('old evidence',encoding='utf-8')
    else:path.mkdir();(path/'result.json').write_text('old evidence',encoding='utf-8')
    before={str(p.relative_to(tmp_path)):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    with pytest.raises(ValueError,match='saved geometry legacy_v1'):
        ensure_geometry_output(path,version,file=file)
    assert before=={str(p.relative_to(tmp_path)):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    marker=ensure_geometry_output(path,'legacy_v1',file=file)
    assert json.loads(marker.read_text())['inherited_existing_artifacts']


@pytest.mark.parametrize('version',GEOMETRY_VERSIONS)
def test_new_results_are_versioned_and_resume_only_same_geometry(tmp_path,version):
    path=tmp_path/'new';marker=ensure_geometry_output(path,version)
    before=marker.read_bytes()
    assert ensure_geometry_output(path,version)==marker and marker.read_bytes()==before
    other=next(v for v in GEOMETRY_VERSIONS if v!=version)
    with pytest.raises(ValueError):ensure_geometry_output(path,other)
    assert marker.read_bytes()==before


@pytest.mark.parametrize('version',['pixel_centers_v2','resize_v2'])
def test_checkpoint_and_reference_geometry_are_checked_read_only(tmp_path,version):
    with pytest.raises(ValueError):require_arm_geometry({'core':{}},version)
    require_arm_geometry({'core':{}},'legacy_v1')
    require_arm_geometry({'core':{'geometry_version':version}},version)
    other=next(v for v in GEOMETRY_VERSIONS if v!=version)
    with pytest.raises(ValueError):require_arm_geometry({'core':{'geometry_version':other}},version)
    with pytest.raises(ValueError):require_reference_geometry(tmp_path,version)
    assert list(tmp_path.iterdir())==[]
    require_reference_geometry(tmp_path,'legacy_v1')


@pytest.mark.parametrize('version',['pixel_centers_v2','resize_v2'])
def test_wide_field_refuses_legacy_cache_and_records_new_cache(tmp_path,monkeypatch,version):
    import scripts.fly_wide_field as wide
    monkeypatch.setattr(wide,'OUT',tmp_path)
    (tmp_path/'cache').mkdir();p=tmp_path/'cache/vid.npz'
    values=(np.zeros((3,2)),np.zeros((2,3,2)),np.zeros((2,3,8)))
    np.savez(p,points=values[0],dec=values[1],t45=values[2]);before=p.read_bytes()
    fly=SimpleNamespace(base=SimpleNamespace(geometry_version=version))
    with pytest.raises(ValueError,match='Cache geometry'):wide.cached(fly,'vid',None)
    assert p.read_bytes()==before
    fly.base.geometry_version='legacy_v1'
    assert all(np.array_equal(a,b) for a,b in zip(wide.cached(fly,'vid',None),values))
    fly.base.geometry_version=version
    monkeypatch.setattr(wide,'video_features',lambda *a:values)
    wide.cached(fly,'new',None)
    with np.load(tmp_path/'cache/new.npz') as z:
        assert z['geometry_version'].item()==version
    assert all(np.array_equal(a,b) for a,b in zip(wide.cached(fly,'new',None),values))


class ModelConstructionReached(Exception):pass


CALLERS=[
    ('fly_axes_probe',True,[]),('fly_dynamic_norm_probe',False,[]),
    ('fly_k1_v2',False,[]),('fly_finetune_k1_synthetic',True,[]),
    ('fly_reference_probe',False,[]),('fly_scale_probe',False,['--synthetic']),
    ('fly_stage_probe',False,[]),('fly_wide_field',False,[]),
    ('build_fly_lcn_motion',False,[]),('fly_noise_probe',True,[]),
    ('fly_speed_response',True,[]),
]
HISTORICAL_GAIN_CALLERS={'fly_k1_v2','fly_finetune_k1_synthetic','fly_wide_field'}


def prepare_entrypoint(module_name,tmp_path,monkeypatch):
    import scripts.fly_finetune_k1 as K
    module=importlib.import_module('scripts.'+module_name)
    monkeypatch.setattr(module,'PROJECT_ROOT',tmp_path)
    monkeypatch.setattr(K,'OUT',tmp_path/'original_k1')
    tc=tmp_path/'train_cache';monkeypatch.setattr(K,'TC',tc)
    fps=tc/'0001542f-5ce3cf52/motion/dis_quarter.json'
    fps.parent.mkdir(parents=True);fps.write_text('{"fps":30.0}',encoding='utf-8')
    (tc/'selection.json').write_text(json.dumps({'videos':{'train':['vid'],'dev':['vid']}}),encoding='utf-8')
    monkeypatch.setattr(K,'read_gray',lambda path:iter([np.full((8,8),i,np.uint8) for i in range(4)]))
    if module_name=='fly_wide_field':
        monkeypatch.setattr(module,'split_videos',lambda:(['vid'],['vid'],{'vid':'night'}))
    if module_name=='build_fly_lcn_motion':
        fp=tmp_path/'runs/video_rate'/module.VAL7[0]/'motion/dis_quarter.json'
        fp.parent.mkdir(parents=True);fp.write_text('{"fps":30.0}',encoding='utf-8')
    return module,K


@pytest.mark.parametrize('module_name,file,extra',CALLERS)
@pytest.mark.parametrize('version',[None,'legacy_v1','pixel_centers_v2','resize_v2'])
def test_all_secondary_entrypoints_forward_version_before_model_run(module_name,file,extra,version,tmp_path,monkeypatch):
    module,K=prepare_entrypoint(module_name,tmp_path,monkeypatch)
    target=tmp_path/('result.json' if file else 'results')
    argv=[module_name,'--out',str(target),*extra]
    if version is not None:argv+=['--geometry-version',version]
    expected=version or 'legacy_v1'
    if module_name=='fly_wide_field':
        reference=tmp_path/'reference';ensure_geometry_output(reference,expected)
        path=reference/'real.json';path.write_text('{"2x3s1_lcn24":{}}',encoding='utf-8')
        argv+=['--reference-results',str(path)]
    seen=[]
    def stop(*args,**kwargs):
        seen.append((kwargs['geometry_version'],kwargs.get('gain')));raise ModelConstructionReached()
    monkeypatch.setattr(K,'Fly',stop)
    monkeypatch.setattr('sys.argv',argv)
    if version=='resize_v2' and module_name in HISTORICAL_GAIN_CALLERS:
        before={str(p.relative_to(tmp_path)):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
        with pytest.raises(ValueError,match='historical fixed gain'):module.main()
        assert seen==[]
        assert before=={str(p.relative_to(tmp_path)):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
        return
    with pytest.raises(ModelConstructionReached):module.main()
    assert len(seen)==1 and seen[0][0]==expected
    if version=='resize_v2':assert seen[0][1]==1.0
    # No network, optimizer, or scientific data calculation was performed.
    markers=list(tmp_path.rglob('*geometry*.json'))
    assert markers and all(json.loads(p.read_text())['geometry_version']==expected for p in markers)


def test_secondary_default_preserves_empirical_vectors_and_lists_three_versions():
    import argparse
    parser=argparse.ArgumentParser();add_geometry_arguments(parser)
    assert GEOMETRY_VERSION=='legacy_v1'
    assert set(GEOMETRY_VERSIONS)=={'legacy_v1','pixel_centers_v2','resize_v2'}
    assert parser.parse_args([]).geometry_version=='legacy_v1'


def test_default_k1_eval_accepts_complete_legacy_arms_without_reinterpreting(tmp_path,monkeypatch):
    module,K=prepare_entrypoint('fly_k1_v2',tmp_path,monkeypatch)
    output=tmp_path/'old_run';output.mkdir()
    checkpoint=output/'arms.pt'
    torch.save({name:{'decoder':{},'network':None} for name in module.ARMS},checkpoint)
    before=checkpoint.read_bytes();seen=[]
    def stop(*args,**kwargs):
        seen.append(kwargs['geometry_version']);raise ModelConstructionReached()
    monkeypatch.setattr(K,'Fly',stop)
    monkeypatch.setattr('sys.argv',['k1','--out',str(output),'--eval-only'])
    with pytest.raises(ModelConstructionReached):module.main()
    assert seen==['legacy_v1'] and checkpoint.read_bytes()==before


@pytest.mark.parametrize('case',['missing','legacy','existing_training','partial_new'])
def test_k1_v2_refuses_invalid_eval_or_overwrite_before_model_or_output_changes(case,tmp_path,monkeypatch):
    import scripts.fly_k1_v2 as module
    import scripts.fly_finetune_k1 as K
    monkeypatch.setattr(module,'PROJECT_ROOT',tmp_path)
    output=tmp_path/'run';output.mkdir()
    if case=='partial_new':torch.save({'decoder':{'geometry_version':'resize_v2'}},output/'arms.pt')
    elif case!='missing':torch.save({'core':{'decoder':{},'network':None}},output/'arms.pt')
    before={p.name:p.read_bytes() for p in output.iterdir()}
    def fail(*args,**kwargs):raise AssertionError('model construction before rejection')
    monkeypatch.setattr(K,'Fly',fail)
    monkeypatch.setattr('sys.argv',['k1','--out',str(output),'--geometry-version','pixel_centers_v2']+([] if case=='existing_training' else ['--eval-only']))
    with pytest.raises((FileNotFoundError,FileExistsError,ValueError)):module.main()
    assert before=={p.name:p.read_bytes() for p in output.iterdir()}


def test_chunked_secondary_runner_accepts_axis_scales_without_scalar_broadcast(monkeypatch):
    import scripts.fly_scale_probe as scale
    n=4;tiles=3
    class Decoder:
        def __call__(self,activity):return torch.stack((activity,-2*activity),dim=2)
    class Fly:
        torch=torch;dev=torch.device('cpu');base=SimpleNamespace(decoder=Decoder())
        def points_scale(self):return np.zeros((tiles*n,2)),np.array([[1,2],[3,4],[5,6]],np.float32)
        def render(self,cart):return cart.mean((-2,-1),keepdim=False)[:,:,None].expand(-1,-1,n)
        def fade_in(self,cur):return SimpleNamespace(nodes=SimpleNamespace(activity=cur[:,0]))
        def step(self,state,last,cur):return self.fade_in(cur),cur
    frames=[np.full((tiles,2,2),v,np.float32) for v in (0,255)]
    output=scale.run_frames(Fly(),iter(frames),chunk=2)
    expected=np.stack([np.tile([sx,2*sy],(n,1)) for sx,sy in ((1,2),(3,4),(5,6))]).reshape(-1,2)
    assert output[0] is None
    np.testing.assert_array_equal(output[1],expected)
