"""Время, научная совместимость и реальная диспетчеризация нового пути Б1."""
import numpy as np
import pytest
import torch
from types import SimpleNamespace

from tracking.b1_timestamped import (
    TimingPolicy, TimestampedBackbone, TimestampedB1Tracker, TimestampedResidualEngine,
    forward_timestamped_residual, load_timestamped_observation,
    make_timestamped_modules, timestamped_checkpoint, restore_timestamped_checkpoint,
    timestamped_pair_activities,
)
from tracking.b1_learning import tensors_for_row
from tracking.flyvis_temporal_residual import causal_pair_activities
from tracking.b1_tracker_runtime import CausalMasa
from tracking.z13_readout import HexGeometry
from tracking.offline_association import _DetSlice


def modules(arm="R1"):
    nb=np.tile(np.arange(4)[:,None],(1,7))
    return make_timestamped_modules(arm,device="cpu",neighbors=nb)


@pytest.mark.parametrize('arm',['R1','B'])
def test_factory_explicit_cpu_does_not_construct_on_ambient_default_device(arm):
    expected_bb,expected_head=modules(arm)
    with torch.device('meta'):
        actual_bb,actual_head=modules(arm)
    assert all(p.device.type=='cpu' for p in actual_head.parameters())
    assert all(torch.equal(v,actual_head.state_dict()[k]) for k,v in expected_head.state_dict().items())
    if arm=='B':
        assert all(p.device.type=='cpu' for p in actual_bb.backend.parameters())
        assert all(torch.equal(v,actual_bb.backend.state_dict()[k]) for k,v in expected_bb.backend.state_dict().items())


def test_device_alias_resolves_current_cuda_index_without_model_allocation(monkeypatch):
    from tracking.b1_timestamped import _concrete_device
    monkeypatch.setattr(torch.cuda,'current_device',lambda:2)
    assert torch.device('cuda')!=torch.device('cuda:2')
    assert _concrete_device('cuda')==torch.device('cuda:2')
    assert _concrete_device('cuda:1')==torch.device('cuda:1')
    assert _concrete_device('cpu:0')==torch.device('cpu')


def test_cpu_factory_preserves_rng_and_does_not_seed_cuda(monkeypatch):
    before=torch.random.get_rng_state().clone()
    def fail(*args,**kwargs):raise AssertionError('CPU factory touched CUDA RNG')
    monkeypatch.setattr(torch.cuda,'manual_seed_all',fail)
    modules('B')
    assert torch.equal(torch.random.get_rng_state(),before)


@pytest.mark.parametrize("arm",["R1","B"])
def test_adjacent_observations_exactly_preserve_old_control_outputs(arm):
    bb,_=modules(arm)
    x=torch.rand(1,5,4)
    ref,cur=causal_pair_activities(bb.backend,x,3)
    fixed_ref,fixed_cur=timestamped_pair_activities(bb,x,[0,1,2,3,4],30.,3)
    assert torch.equal(ref,fixed_ref) and torch.equal(cur,fixed_cur)


@pytest.mark.parametrize("arm",["R1","B"])
def test_skipped_frames_equal_explicit_linear_observations(arm):
    bb,_=modules(arm)
    x=torch.rand(1,3,4)
    expanded=torch.stack([x[:,0],x[:,0]*.5+x[:,1]*.5,x[:,1],
                          x[:,1]*(2/3)+x[:,2]*(1/3),x[:,1]*(1/3)+x[:,2]*(2/3),x[:,2]],1)
    _,actual=bb.sequence(x,[18,20,23],30.)
    _,expected=bb.sequence(expanded,[18,19,20,21,22,23],30.)
    assert torch.allclose(actual[-1],expected[-1],atol=2e-7,rtol=1e-6)


@pytest.mark.parametrize("arm",["R1","B"])
def test_padded_duplicate_frames_consume_zero_time(arm):
    bb,_=modules(arm)
    x=torch.rand(1,2,4)
    padded=torch.cat((x[:,:1].expand(-1,4,-1),x[:,1:]),1)
    _,expected=bb.sequence(x,[24,26],30.)
    _,actual=bb.sequence(padded,[24,24,24,24,26],30.)
    assert torch.equal(actual[-1],expected[-1])
    assert all(torch.equal(actual[0],a) for a in actual[:4])


def test_fly_interval_uses_real_duration_and_restores_no_global_clock():
    calls=[]
    class Net:
        stimulus=SimpleNamespace(input_index=np.array([[0,1]]),n_nodes=2)
        def forward(self,values,dt,state,as_states):
            calls.append((values.clone(),dt))
            return [SimpleNamespace(nodes=SimpleNamespace(activity=values[:,-1]))]
    raw=SimpleNamespace(dev=torch.device('cpu'),net=Net(),index=torch.tensor([[0,1]]),steps=2,dt=1/60)
    bb=TimestampedBackbone(raw,'C')
    x=torch.tensor([[0.,1.]])
    bb.step(None,x,1-x,previous_frame=18,current_frame=21,source_fps=30.)
    values,dt=calls[0]
    assert values.shape[1]==5 and dt==pytest.approx(.02)
    assert values.shape[1]*dt==pytest.approx(.1)
    assert raw.steps==2 and raw.dt==1/60
    assert torch.equal(values[:,-1],1-x)


@pytest.mark.parametrize("frames,fps",[([2,1,3],30),([0,1,1.5],30),([0,True,2],30),([0,1,2],0),([0,1,2],float('nan'))])
def test_invalid_timestamps_rejected(frames,fps):
    bb,_=modules()
    with pytest.raises(ValueError):bb.sequence(torch.ones(1,3,4),frames,fps)


def test_duplicate_time_with_different_input_rejected():
    bb,_=modules()
    with pytest.raises(ValueError,match='Repeated frame'):
        bb.sequence(torch.tensor([[[0.,0.,0.,0.],[1.,1.,1.,1.]]]),[3,3],30.)


def test_invalid_policy_and_reference_rejected():
    for value in (0,-1,.03,float('nan')):
        with pytest.raises(ValueError):TimingPolicy(value)
    bb,_=modules()
    with pytest.raises(ValueError):timestamped_pair_activities(bb,torch.ones(1,3,4),[3,3,3],30.,1)


@pytest.mark.parametrize("arm",["R1","B"])
def test_training_and_incremental_runtime_match_with_nonzero_head(arm):
    bb,head=modules(arm)
    torch.nn.init.normal_(head.output.weight, std=.1)
    x=torch.rand(1,6,4);x[:,1]=x[:,0];frames=[18,18,21,24,26,29]
    row=dict(reference_index=3,choices=[-1,0,1],current_masks=[[0,1],[2,3]],
             reference_mask=[0,1],has_history=True)
    p=np.array([np.nan,.2,.7],np.float64)
    inputs=tensors_for_row(row,x,p,'cpu')
    expected,_=forward_timestamped_residual(bb,head,x,row,inputs,
        frame_numbers=frames,source_fps=30.,train_backbone=True)
    ck=timestamped_checkpoint(bb,head,cursor=1)
    live_bb,live_head=modules(arm)
    engine=TimestampedResidualEngine(live_bb,live_head,ck,source_fps=30.)
    eye=engine.start([dict(frame=f,hex=h) for f,h in zip(frames[:4],x[0,:4])])
    for frame,h in zip(frames[4:],x[0,4:]):engine.advance_at(eye,h,frame)
    actual,_=engine.correction(eye,inputs['masks'][0].numpy(),inputs['reference_mask'][0].numpy(),p)
    assert np.array_equal(actual,expected.detach().numpy()[0],equal_nan=True)
    zero,_=engine.correction(eye,inputs['masks'][0].numpy(),inputs['reference_mask'][0].numpy(),p,zero=True)
    assert np.array_equal(zero,p,equal_nan=True)
    with pytest.raises(ValueError):engine.advance_at(eye,x[0,-1],frames[-1])


def test_old_checkpoint_rejected_before_any_weight_changes():
    bb,head=modules('B');ck=timestamped_checkpoint(bb,head,cursor=2)
    before={k:v.clone() for k,v in head.state_dict().items()}
    del ck['temporal_semantics']
    with pytest.raises(ValueError,match='Legacy'):restore_timestamped_checkpoint(ck,bb,head)
    assert all(torch.equal(v,before[k]) for k,v in head.state_dict().items())


def test_diagnostic_checkpoint_cannot_be_promoted_by_normal_runtime():
    bb,head=modules();ck=timestamped_checkpoint(bb,head,cursor=2,
        provenance=dict(diagnostic_only=True))
    before={k:v.clone() for k,v in head.state_dict().items()}
    with pytest.raises(ValueError,match='Engineering smoke'):
        TimestampedResidualEngine(bb,head,ck,source_fps=30.)
    assert head.training
    assert all(p.requires_grad for p in head.parameters())
    assert all(torch.equal(v,before[k]) for k,v in head.state_dict().items())
    engine=TimestampedResidualEngine(bb,head,ck,source_fps=30.,allow_diagnostic=True)
    assert engine.cursor==2 and not head.training


def test_checkpoint_is_snapshot_and_restores_optimizer(tmp_path):
    bb,head=modules('B');opt=torch.optim.Adam(list(head.parameters())+list(bb.backend.parameters()))
    sum(v.sum() for v in head.parameters()).backward();opt.step()
    ck=timestamped_checkpoint(bb,head,cursor=3,optimizer=opt)
    expected={k:v.clone() for k,v in ck['head'].items()}
    with torch.no_grad():next(head.parameters()).add_(5)
    assert all(torch.equal(v,expected[k]) for k,v in ck['head'].items())
    path=tmp_path/'new_time.pt';torch.save(ck,path)
    assert restore_timestamped_checkpoint(torch.load(path,weights_only=False),bb,head,optimizer=opt)==3
    assert all(torch.equal(v,expected[k]) for k,v in head.state_dict().items())


def test_checkpoint_validates_both_parts_before_copying_weights():
    bb,head=modules('B');ck=timestamped_checkpoint(bb,head,cursor=1)
    before={k:v.clone() for k,v in bb.backend.state_dict().items()}
    ck['backbone']['zr.w'].add_(1)
    del ck['head']['output.weight']
    with pytest.raises(ValueError,match='architecture'):restore_timestamped_checkpoint(ck,bb,head)
    assert all(torch.equal(v,before[k]) for k,v in bb.backend.state_dict().items())


@pytest.mark.parametrize('cursor',[True,-1,1.5])
def test_checkpoint_cursor_does_not_silently_coerce_values(cursor):
    bb,head=modules()
    with pytest.raises(ValueError,match='cursor'):timestamped_checkpoint(bb,head,cursor=cursor)
    ck=timestamped_checkpoint(bb,head,cursor=1);ck['cursor']=cursor
    with pytest.raises(ValueError,match='cursor'):restore_timestamped_checkpoint(ck,bb,head)


def test_float_cast_overflow_rejected_before_backbone_changes():
    bb,head=modules('B');ck=timestamped_checkpoint(bb,head,cursor=1)
    before={k:v.clone() for k,v in bb.backend.state_dict().items()}
    ck['backbone']['zr.w'].add_(1)
    ck['head']['output.weight']=torch.full_like(ck['head']['output.weight'],1e100,dtype=torch.float64)
    with pytest.raises(ValueError):restore_timestamped_checkpoint(ck,bb,head)
    assert all(torch.equal(v,before[k]) for k,v in bb.backend.state_dict().items())


def test_integer_connectivity_cannot_be_loaded_from_fractional_values():
    bb,head=modules('B');ck=timestamped_checkpoint(bb,head,cursor=1)
    ck['backbone']['zr.nb']=ck['backbone']['zr.nb'].float()+.5
    with pytest.raises(ValueError,match='dtype'):restore_timestamped_checkpoint(ck,bb,head)


def test_checkpoint_cannot_replace_runtime_receptor_geometry():
    bb,head=modules('B');ck=timestamped_checkpoint(bb,head,cursor=1)
    ck['backbone']['zr.nb'][0,0]=1
    with pytest.raises(ValueError,match='geometry'):restore_timestamped_checkpoint(ck,bb,head)


@pytest.mark.parametrize('defect',['group_count','nonfinite','shape','learning_rate'])
def test_bad_optimizer_state_rejected_before_model_changes(defect):
    bb,head=modules('B');opt=torch.optim.Adam(head.parameters())
    sum(v.sum() for v in head.parameters()).backward();opt.step()
    ck=timestamped_checkpoint(bb,head,cursor=1,optimizer=opt)
    before={k:v.clone() for k,v in head.state_dict().items()}
    ck['head']['output.weight'].add_(10)
    if defect=='group_count':ck['optimizer']['param_groups']=[]
    elif defect=='learning_rate':ck['optimizer']['param_groups'][0]['lr']=float('nan')
    else:
        state=next(iter(ck['optimizer']['state'].values()))
        state['exp_avg']=torch.full_like(state['exp_avg'],float('inf')) if defect=='nonfinite' else torch.zeros(999)
    with pytest.raises(ValueError):restore_timestamped_checkpoint(ck,bb,head,optimizer=opt)
    assert all(torch.equal(v,before[k]) for k,v in head.state_dict().items())


def test_failure_during_actual_optimizer_load_rolls_back_model_and_optimizer():
    bb,head=modules('B');opt=torch.optim.Adam(head.parameters())
    sum(v.sum() for v in head.parameters()).backward();opt.step()
    ck=timestamped_checkpoint(bb,head,cursor=1,optimizer=opt)
    before=timestamped_checkpoint(bb,head,cursor=1,optimizer=opt)
    ck['head']['output.weight'].add_(10);ck['backbone']['zr.w'].add_(1)
    def fail_only_live(loading_opt, state):
        if loading_opt is opt:
            loading_opt.param_groups[0]['lr']=123.
            raise RuntimeError('live optimizer load failed')
    handle=opt.register_load_state_dict_pre_hook(fail_only_live)
    with pytest.raises(RuntimeError,match='live optimizer'):
        restore_timestamped_checkpoint(ck,bb,head,optimizer=opt)
    handle.remove()
    assert all(torch.equal(v,before['head'][k]) for k,v in head.state_dict().items())
    assert all(torch.equal(v,before['backbone'][k]) for k,v in bb.backend.state_dict().items())
    after=opt.state_dict()
    assert after['param_groups']==before['optimizer']['param_groups']
    for key,state in after['state'].items():
        assert all(torch.equal(value,before['optimizer']['state'][key][name]) for name,value in state.items())
    # Restored state still belongs to the live Parameters and supports the next step.
    opt.zero_grad();sum(v.sum() for v in head.parameters()).backward();opt.step()


def test_archive_time_binding_and_path_boundary(tmp_path):
    np.savez(tmp_path/'eye.npz',fr_hex=np.ones((5,4),np.float16),fr_frame=np.array([18,21,22,24,26]))
    row=dict(retinal_source='eye.npz',prefix_start=0,current_array_index=4,reference_index=3,
             reference_frame=24,current_frame=26)
    hx,frames=load_timestamped_observation(tmp_path,row,'cpu')
    assert frames.tolist()==[18,21,22,24,26] and hx.shape==(1,5,4)
    with pytest.raises(ValueError):load_timestamped_observation(tmp_path,dict(row,current_frame=25),'cpu')
    with pytest.raises(ValueError):load_timestamped_observation(tmp_path,dict(row,retinal_source='../escape.npz'),'cpu')


def test_real_tracker_passes_two_frame_loss_interval_to_engine():
    class Bank:
        eye_size=391
        def _render(self,frame,crops):return torch.full((len(crops),1,4),float(frame[0,0])/10)
    class Server(CausalMasa):
        def embed(self,frame,boxes):return np.zeros((len(boxes),256),np.float32)
    class Base:
        def probability(self,geo,region,coarse,snapped,old,gap):
            return np.full(len(geo),.5),geo,np.ones(len(geo),bool),{},np.zeros((len(geo),1))
        def decision(self,p,geo,mask):return None,np.full(len(geo),.5)
    bb,head=modules();ck=timestamped_checkpoint(bb,head,cursor=0)
    engine=TimestampedResidualEngine(bb,head,ck,source_fps=30.)
    calls=[];original=bb.step
    def watched(*a,**kw):
        calls.append((kw['previous_frame'],kw['current_frame']))
        return original(*a,**kw)
    bb.step=watched
    geom=HexGeometry(np.array([[160,160],[230,160],[160,230],[230,230]],np.float32),391)
    args=SimpleNamespace(track_high_thresh=.25,track_low_thresh=.1,new_track_thresh=.25,track_buffer=30,match_thresh=.8,fuse_score=True)
    server=Server(None);tracker=TimestampedB1Tracker(args,Bank(),geom,server,Base(),engine)
    for frame in range(7):
        boxes=np.array([[30,30,70,70]],np.float32) if frame<4 else np.zeros((0,4),np.float32)
        conf=np.full(len(boxes),.9,np.float32);gray=np.full((100,100),frame,np.uint8)
        server.start(frame,gray,boxes,conf);tracker.begin(frame,gray)
        tracker.update(_DetSlice(boxes,conf,np.zeros(len(boxes),np.float32)));tracker.after()
    assert (3,5) in calls and (5,6) in calls
    with pytest.raises(ValueError):tracker.begin(6,gray)


def test_reset_clears_private_history_and_preserves_other_trackers_counter():
    from ultralytics.trackers.basetrack import BaseTrack
    args=SimpleNamespace(track_high_thresh=.25,track_low_thresh=.1,new_track_thresh=.25,track_buffer=30,match_thresh=.8,fuse_score=True)
    saved=BaseTrack._count
    try:
        BaseTrack._count=456
        tracker=TimestampedB1Tracker(args,disabled=True)
        assert BaseTrack._count==456
        tracker.hist[1]='old';tracker.eyes[1]='old';tracker.previous[1]='old';tracker.events.append('old')
        det=_DetSlice(np.array([[30,30,70,70]],np.float32),np.array([.9],np.float32),np.array([0],np.float32))
        gray=np.zeros((100,100),np.uint8)
        tracker.begin(8,gray);first=tracker.update(det)
        tracker.reset()
        assert BaseTrack._count==456
        assert not any((tracker.hist,tracker.eyes,tracker.previous,tracker.events))
        assert tracker._observed_frame is None
        tracker.begin(0,gray);second=tracker.update(det)
        fresh=TimestampedB1Tracker(args,disabled=True);fresh.begin(0,gray)
        assert np.array_equal(first,second) and np.array_equal(second,fresh.update(det))
        assert BaseTrack._count==456
    finally:BaseTrack._count=saved


@pytest.mark.parametrize('arm',['R1','B'])
@pytest.mark.parametrize('defect',['state_nan','activity_nan','state_shape','activity_shape'])
def test_failed_backend_transition_does_not_publish_eye_or_advance_clock(arm,defect):
    bb,head=modules(arm);checkpoint=timestamped_checkpoint(bb,head,cursor=0)
    engine=TimestampedResidualEngine(bb,head,checkpoint,source_fps=30.)
    x=torch.tensor([[.1,.2,.3,.4],[.15,.25,.35,.45],[.2,.3,.4,.5]],dtype=torch.float32)
    history=[dict(frame=i,hex=h) for i,h in enumerate(x[:2])]
    eye=engine.start(history);control=engine.start(history)
    previous=dict(eye);before_state=eye['state'].clone() if arm=='R1' else eye['state']['h'].clone()
    raw_step=bb.backend.step
    def broken(*args):
        state,activity=raw_step(*args)
        if defect=='state_nan':
            if arm=='R1':state=torch.full_like(state,float('nan'))
            else:state={'h':torch.full_like(state['h'],float('nan'))}
        elif defect=='activity_nan':activity=torch.full_like(activity,float('nan'))
        elif defect=='state_shape':state=state[:,:,:-1] if arm=='R1' else {'h':state['h'][:,:,:-1]}
        else:activity=activity[:,:,:-1]
        return state,activity
    bb.backend.step=broken
    with pytest.raises(ValueError):engine.advance_at(eye,x[2],2)
    assert eye.keys()==previous.keys() and all(eye[k] is previous[k] for k in previous)
    assert torch.equal(before_state,eye['state'] if arm=='R1' else eye['state']['h'])
    bb.backend.step=raw_step
    engine.advance_at(eye,x[2],2);engine.advance_at(control,x[2],2)
    assert eye['last_frame']==2
    assert torch.equal(eye['current'],control['current'])
    assert torch.equal(eye['state'] if arm=='R1' else eye['state']['h'],
                       control['state'] if arm=='R1' else control['state']['h'])


@pytest.mark.parametrize('arm',['R1','B'])
def test_nonfinite_initial_state_is_rejected_before_history_is_published(arm):
    bb,head=modules(arm);checkpoint=timestamped_checkpoint(bb,head,cursor=0)
    engine=TimestampedResidualEngine(bb,head,checkpoint,source_fps=30.)
    original=bb.backend.init
    def broken(first):
        state=original(first)
        return torch.full_like(state,float('nan')) if arm=='R1' else {'h':torch.full_like(state['h'],float('nan'))}
    bb.backend.init=broken
    with pytest.raises(ValueError):engine.start([dict(frame=0,hex=torch.ones(4))])


def test_finite_receptor_values_that_overflow_real_r1_state_are_rejected():
    bb,head=modules();engine=TimestampedResidualEngine(bb,head,timestamped_checkpoint(bb,head,cursor=0),source_fps=30.)
    large=torch.full((4,),torch.finfo(torch.float32).max)
    eye=engine.start([dict(frame=0,hex=large)])
    before=dict(eye)
    with pytest.raises(ValueError,match='finite'):engine.advance_at(eye,-large,1)
    assert all(eye[k] is before[k] for k in before)
    engine.advance_at(eye,large,1)
    assert eye['last_frame']==1 and torch.isfinite(eye['state']).all()


def test_nonfinite_unread_core_neuron_is_rejected_even_with_finite_readout():
    class Net:
        stimulus=SimpleNamespace(input_index=np.array([[0,1]]),n_nodes=3)
        def forward(self,values,dt,state,as_states):
            activity=torch.tensor([[1.,2.,float('nan')]])
            return [SimpleNamespace(nodes=SimpleNamespace(activity=activity)) for _ in range(values.shape[1])]
    raw=SimpleNamespace(dev=torch.device('cpu'),net=Net(),index=torch.tensor([[0,1]]))
    bb=TimestampedBackbone(raw,'C');x=torch.ones(1,2)
    with pytest.raises(ValueError,match='finite'):
        bb.step(None,x,x,previous_frame=0,current_frame=1,source_fps=30.)


def test_finite_extreme_activity_cannot_publish_nonfinite_head_probability():
    bb,head=modules();engine=TimestampedResidualEngine(bb,head,timestamped_checkpoint(bb,head,cursor=0),source_fps=30.)
    large=torch.full((4,),torch.finfo(torch.float32).max)
    eye=engine.start([dict(frame=0,hex=large)]);engine.advance_at(eye,large,1)
    assert torch.isfinite(eye['current']).all()
    with pytest.raises(ValueError,match='finite'):
        engine.correction(eye,np.ones((2,4),bool),np.ones(4,bool),np.array([np.nan,.5]))
    assert engine.last_pooled is None


def test_training_uses_the_same_nonfinite_head_boundary():
    bb,head=modules();x=torch.full((1,3,4),torch.finfo(torch.float32).max)
    row=dict(reference_index=1,choices=[-1,0],current_masks=[[0,1,2,3]],reference_mask=[0,1,2,3],has_history=True)
    inputs=tensors_for_row(row,x,np.array([np.nan,.5]),'cpu')
    with pytest.raises(ValueError,match='finite'):
        forward_timestamped_residual(bb,head,x,row,inputs,frame_numbers=[0,1,2],source_fps=30.,train_backbone=False)
