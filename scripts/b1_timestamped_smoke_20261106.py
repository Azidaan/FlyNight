"""Исполняемая CPU-проверка новой временной версии на сохранённом входе Б1.

Пример: .venv/Scripts/python.exe scripts/b1_timestamped_smoke_20261106.py --arm D
Новый каталог результатов; не продолжает и не перезаписывает старое обучение.
Два технических шага оптимизатора проверяют производные и сохранение, не HOTA.
Контрольная точка помечена diagnostic_only и запрещена обычным исполнителем.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone, timedelta

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',choices=('C','D','R1','B'),default='D')
    parser.add_argument('--output',type=Path,default=ROOT/'runs/flyvis_timestamp_repair_20261106/real_D')
    args=parser.parse_args()
    out=args.output.resolve();out.relative_to(ROOT)
    out.mkdir(parents=True,exist_ok=False)
    os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ['MPLBACKEND']='Agg'
    selection=ROOT/'runs/b1_real_inputs_20261026/gpu_samples.json'
    row=json.loads(selection.read_text(encoding='utf-8'))['rows'][0]
    fps_file=ROOT/f'runs/train_cache/{row["video"]}/motion/flyvis_tiled.json'
    fps=float(json.loads(fps_file.read_text(encoding='utf-8'))['fps'])
    files=[Path(__file__),ROOT/'tracking/b1_timestamped.py',ROOT/'tracking/flyvis_checkpoint.py',ROOT/'tracking/b1_learning.py',
           ROOT/'tracking/b1_tracker_runtime.py',ROOT/'tracking/z13_readout.py',ROOT/'tracking/flyvis_temporal_residual.py',
           selection,ROOT/row['retinal_source'],fps_file]
    protocol=dict(registered_msk=datetime.now(timezone(timedelta(hours=3))).isoformat(),arm=args.arm,uid=row['uid'],source_fps=fps,
        scope='CPU engineering validation only. One previously saved row, two synthetic identity-loss optimizer steps; no scientific training result or HOTA.',
        criteria=['Timestamped adjacent pair agrees with original backend at same FPS',
                  'All gaps use saved frame numbers; duplicate IDs consume no time',
                  'Zero head preserves base exactly; two optimizer steps have finite gradients',
                  'Checkpoint save/load and streamed inference reproduce offline corrected probability exactly',
                  'Old sources and original data unchanged; diagnostic checkpoint rejected by default inference'],
        sha256={str(p.relative_to(ROOT)):sha(p) for p in files})
    write(out/'protocol.json',protocol)
    import numpy as np
    import torch
    torch.set_num_threads(2);torch.set_default_device('cpu')
    from tracking.b1_timestamped import (make_timestamped_modules,forward_timestamped_residual,
        load_timestamped_observation,timestamped_checkpoint,restore_timestamped_checkpoint,TimestampedResidualEngine)
    from tracking.b1_learning import tensors_for_row,identity_objective
    estimator=None
    if args.arm in ('C','D','B'):
        import importlib
        from tracking.flyvis_adapter import activate_flyvis
        fv=activate_flyvis()
        for module in (fv,importlib.import_module('flyvis.network.initialization'),importlib.import_module('flyvis.task.decoder')):
            module.device=torch.device('cpu')
        torch.set_default_device('cpu')
        from tracking.flyvis_motion import FlyVisMotionEstimator
        estimator=FlyVisMotionEstimator(source_fps=fps,flow_gain=1.)
        from tracking.foveated_eyes import EyeBank
        from tracking.z13_readout import HexGeometry
        bank=EyeBank(estimator);geom=HexGeometry(bank.points_local,bank.eye_size)
        neighbors=geom.neighbors
    else:
        neighbors=None
    bb,head=make_timestamped_modules(args.arm,device='cpu',estimator=estimator,neighbors=neighbors)
    hx,frames=load_timestamped_observation(ROOT,row,'cpu')
    probability=np.linspace(.25,.75,len(row['choices']),dtype=np.float64);probability[0]=np.nan
    inputs=tensors_for_row(row,hx,probability,'cpu')
    with torch.no_grad():
        state,_=bb.init(hx[:,:1])
        _,old=bb.backend.step(state,hx[:,0],hx[:,1])
        _,new=bb.step(state,hx[:,0],hx[:,1],previous_frame=0,current_frame=1,source_fps=fps)
        adjacent_difference=float((old-new).abs().max())
        assert adjacent_difference<=1e-6
    groups=[dict(params=list(head.parameters()),lr=.001)]
    if args.arm=='D':groups+=bb.backend.param_groups()
    elif args.arm=='B':groups.append(dict(params=list(bb.backend.parameters()),lr=.001))
    opt=torch.optim.Adam(groups)
    gradients=[];losses=[]
    labels=torch.ones_like(inputs['base']);labels[:,0]=-1
    for step in range(2):
        opt.zero_grad(set_to_none=True)
        updated,delta=forward_timestamped_residual(bb,head,hx,row,inputs,frame_numbers=frames,
            source_fps=fps,train_backbone=args.arm in ('D','B'))
        if step==0:assert torch.equal(updated[inputs['real']],inputs['base'][inputs['real']])
        loss=identity_objective(updated,delta,labels,inputs['real'])
        if loss is None:raise ValueError('Selected technical sample has no real region')
        loss.backward()
        params=[p for group in opt.param_groups for p in group['params']]
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in params)
        if args.arm=='D':
            gradients.append({name:float(p.grad.norm()) if p.grad is not None else None
                for name,p in bb.backend.net.named_parameters() if p.requires_grad})
        opt.step()
        if args.arm=='D':bb.backend.clamp()
        losses.append(float(loss.detach()))
    if args.arm=='D':assert all(v is not None and v>0 for v in gradients[-1].values())
    with torch.no_grad():
        expected,_=forward_timestamped_residual(bb,head,hx,row,inputs,frame_numbers=frames,
            source_fps=fps,train_backbone=False)
    checkpoint=timestamped_checkpoint(bb,head,cursor=2,optimizer=opt,
        provenance=dict(diagnostic_only=True,protocol_sha256=sha(out/'protocol.json')))
    torch.save(checkpoint,out/'diagnostic_checkpoint.pt')
    checkpoint=torch.load(out/'diagnostic_checkpoint.pt',map_location='cpu',weights_only=False)
    # Prove restore actually restores weights and moments, rather than loading
    # unchanged objects and only checking that deserialization did not raise.
    with torch.no_grad():
        next(head.parameters()).add_(1.)
        if args.arm in ('C','D'):next(bb.backend.net.parameters()).add_(1.)
        elif args.arm=='B':next(bb.backend.parameters()).add_(1.)
    opt.state.clear();opt.param_groups[0]['lr']=.987
    assert restore_timestamped_checkpoint(checkpoint,bb,head,optimizer=opt)==2
    restored_opt=opt.state_dict();saved_opt=checkpoint['optimizer']
    assert restored_opt['param_groups']==saved_opt['param_groups']
    assert set(restored_opt['state'])==set(saved_opt['state'])
    for key,state in restored_opt['state'].items():
        assert all(torch.equal(value,saved_opt['state'][key][name]) for name,value in state.items())
    with torch.no_grad():
        restored,_=forward_timestamped_residual(bb,head,hx,row,inputs,frame_numbers=frames,
            source_fps=fps,train_backbone=False)
    assert np.array_equal(expected.cpu().numpy(),restored.cpu().numpy(),equal_nan=True)
    # The engine takes ownership of this backend and freezes it for inference.
    try:TimestampedResidualEngine(bb,head,checkpoint,source_fps=fps)
    except ValueError:diagnostic_rejected=True
    else:raise AssertionError('Diagnostic checkpoint unexpectedly admitted')
    engine=TimestampedResidualEngine(bb,head,checkpoint,source_fps=fps,allow_diagnostic=True)
    reference=row['reference_index']
    eye=engine.start([dict(frame=int(f),hex=x) for f,x in zip(frames[:reference+1],hx[0,:reference+1])])
    for frame,value in zip(frames[reference+1:],hx[0,reference+1:]):engine.advance_at(eye,value,int(frame))
    actual,_=engine.correction(eye,inputs['masks'][0].numpy(),inputs['reference_mask'][0].numpy(),probability)
    expected=expected.cpu().numpy()[0]
    assert np.array_equal(expected,actual,equal_nan=True)
    np.savez(out/'outputs.npz',expected=expected,actual=actual,frames=frames)
    unchanged={p:sha(ROOT/p)==h for p,h in protocol['sha256'].items()}
    assert all(unchanged.values())
    report=dict(passed=True,arm=args.arm,frames=frames.tolist(),source_fps=fps,
        elapsed_input_seconds=float((frames[-1]-frames[0])/fps),adjacent_max_abs_difference=adjacent_difference,
        losses=losses,core_gradient_norms=gradients,offline_streamed_exact=True,
        checkpoint_restore_after_mutation_exact=True,optimizer_restore_exact=True,
        diagnostic_rejected=diagnostic_rejected,cuda_initialized=torch.cuda.is_initialized(),
        unchanged=unchanged,checkpoint_sha256=sha(out/'diagnostic_checkpoint.pt'),outputs_sha256=sha(out/'outputs.npz'),
        protocol_sha256=sha(out/'protocol.json'),scientific_hypothesis_accepted=False,hota_measured=False)
    write(out/'report.json',report);print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
