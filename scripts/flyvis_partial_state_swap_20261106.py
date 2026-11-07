"""Partial recurrent-state transplants, reversal and continued-motion controls."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse,hashlib,importlib,json,sys,time,shutil
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/flyvis_partial_state_swap_20261106';OLD=ROOT/'runs/flyvis_history_state_swap_20261106';SRC=ROOT/'runs/flyvis_offset_recovery2_20261104'
sys.path.insert(0,str(ROOT))
from scripts.flyvis_history_state_swap_20261106 import sha,save
GROUPS=['retina','intermediate','t4','t5','other_readout','t4t5']
ARMS=['moving','stationary','identity_moving','identity_stationary']+[f'{direction}_{g}' for g in GROUPS for direction in ('erase','insert')]
def journal(text):
    p=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';raw=p.read_bytes();marker='## Журнал изменений'.encode();nl=b'\r\n' if marker+b'\r\n' in raw else b'\n'
    assert raw.count(marker+nl)==1;now=datetime.now(ZoneInfo('Europe/Moscow'))
    p.write_bytes(raw.replace(marker+nl,marker+nl+f'\n**{now:%d.%m.%Y, %H:%M} MSK** — Частичная подмена состояния FlyVis\n\n{text}\n\n'.encode(),1))
def register():
    OUT.mkdir(exist_ok=False)
    old=json.loads((OLD/'protocol.json').read_text(encoding='utf-8'));audit=json.loads((OLD/'independent_audit.json').read_text(encoding='utf-8'));assert audit['passed'] and audit['cases']==32
    connectome=ROOT/'tools/flyvis-runtime/flyvis/connectome/fib25-fib19_v2.2.json';data=json.loads(connectome.read_text());names=[x['name'] for x in data['nodes']]
    groups=dict(retina=data['input_units'],intermediate=[n for n in names if n not in data['input_units']+data['output_units']],t4=['T4a','T4b','T4c','T4d'],t5=['T5a','T5b','T5c','T5d'],other_readout=[n for n in data['output_units'] if n not in ['T4a','T4b','T4c','T4d','T5a','T5b','T5c','T5d']])
    assert sorted(sum(groups.values(),[]))==sorted(names);groups['t4t5']=groups['t4']+groups['t5']
    sources=['scripts/flyvis_partial_state_swap_20261106.py','scripts/audit_flyvis_partial_state_swap_20261106.py','scripts/control_flyvis_partial_state_swap_20261106.py','tracking/flyvis_state_transplant.py','scripts/flyvis_history_state_swap_20261106.py','tracking/flyvis_motion.py','tracking/flyvis_adapter.py','tools/flyvis-runtime/flyvis/network/network.py','tools/flyvis-runtime/flyvis/network/dynamics.py','tools/flyvis-runtime/flyvis/task/decoder.py','tools/flyvis-runtime/flyvis/connectome/fib25-fib19_v2.2.json','scripts/flyvis_y1_latency_fp64_20261104.py','runs/flyvis/results/flow/0000/000/chkpts/chkpt_00000']
    refs={c['id']:dict(state_sha256=sha(OLD/'cases'/(c['id']+'.npz')),input_sha256=sha(SRC/'inputs'/(c['id']+'.npz'))) for c in old['cases']}
    criterion='Technical: exact selected/unselected coordinates and identity controls; full-state reverse controls reproduce old saved outputs <=1e-10; no state/parameter/buffer mutation; all finite and independent audit passed. For each of six fixed groups: ordinary-contrast erase-minus-moving latency upper99.1667% scene-bootstrap bound<0 (six-comparison family); reversal EPE at both contrasts no worse in first6 and all24 frames, and missing latency counts no larger; continued-motion EPE at both contrasts no worse in first6 and all24 frames, and correct-sign fraction no smaller. All conditions required for a preservation candidate; no tuning or automatic deployment.'
    p=dict(id='flyvis_partial_state_swap_v1',registered_msk=datetime.now(ZoneInfo('Europe/Moscow')).isoformat(),deadline_msk='2026-11-07T01:00:00+03:00',cases=old['cases'],groups=groups,arms=ARMS,scenarios=['reverse','continue'],reference_sha256=refs,old_protocol_sha256=sha(OLD/'protocol.json'),old_audit_sha256=sha(OLD/'independent_audit.json'),parameter_sha=json.loads((OLD/'inference_completed.json').read_text())['parameter_sha'],frozen_sources={s:sha(ROOT/s) for s in sources},geometry_sha256=sha(SRC/'geometry.npz'),source_images_sha256=sha(SRC/'source_images.json'),criterion=criterion,
      intervention='Restore saved two full boundary states after frame54 (no recomputation of completed histories). For each group copy stationary donor values into moving state (erase), and moving into stationary (insert), all other coordinates exact. Rebuild derived source/target references. Recurrence, not merely decoder input, is changed. Five disjoint partitions cover all65 types; T4+T5 union prespecified. Groups differ in size/activation: no matched-dose or sole-cause claim. No LPi cells in this model.',
      inputs='Same four previously studied scenes4..7, contrasts1/.25, speeds1/4, axes0/1. Reverse suffix frames55..78 and boundary states reused exactly by SHA. Continued-motion suffix crops original same first image at positions speed*(25..48), preserving same frame54 position speed*24. Source grayscale SHA must match original source_images.json. No new dataset, weights, fitting, or GPU.',
      execution='Stock flow/0000/000 CPU FP64, legacy_v1 input/output convention, hold2 internal steps, last-step decoder, no interpolation or temporal averaging, no normalization/calibration. Official double BoxEye for new continuation; independent OpenCV eye<=1e-5 and boundary-eye exact<=1e-12. All16 arms share suffix, separate states. One-time swap only at boundary before frame55.',
      metrics='Central-mask mean vector EPE in nominal pixels/frame, first6 and all24 frames, correct-sign fraction; reversal first3 consecutive correct sign, absent capped24 with missing count. Scene-cluster bootstrap20k seed42; six forward latency tests99.1667% CI; all remaining intervals95% descriptive. Insert direction checks restoration of delay, does not alone identify biological cause. Four scenes remain exploratory.',
      decision='If no group both reduces reversal latency and preserves continuing motion, no safe memory-removal candidate. Do not redefine criteria or tune neuron groups on these results. If a candidate passes, it remains diagnostic using an artificial donor and requires a feasible causal implementation and new scenes before deployment.',
      literature=[dict(url='https://www.nature.com/articles/s41586-024-07939-3',use='Connectome-constrained recurrent model and separate readout; basis for isolating state rather than scaling only decoder input.'),dict(url='https://doi.org/10.1016/j.cub.2021.09.072',use='T4 and T5 motion/contrast pathways justify separate and joint cell-type groups, not a published solution to this delay.')],
      limitations='Artificial hybrid states may be off the normal trajectory. Better reversal response may destroy useful memory; continued motion is one limited safeguard, not proof of all memory utility. One model/four previously studied TRAIN images, no HOTA/natural video/independent final validation or sole-cell cause.',watchdog_seconds=600,minimum_free_bytes=5_000_000_000)
    save(OUT/'protocol.json',p);save(OUT/'status.json',dict(stage='registered',done=0,total=32))
    cat=ROOT/'experiments/project_history.json';d=json.loads(cat.read_text(encoding='utf-8'));assert not any(c['id']==p['id'] for c in d['experiments'])
    d['experiments'].append(dict(id=p['id'],title='FlyVis — какие части состояния задерживают разворот и сохраняют движение',area='flyvis',state='planned',started='2026-11-06',tags=['Y1','motion-diagnostic'],question='Можно ли локализовать влияние предыстории без ухудшения продолжающегося движения?',hypothesis='Влияние может находиться в отдельных группах или быть распределено; частичная подмена может одновременно уменьшать задержку и повреждать полезный сигнал.',expected='Сравнить шесть заранее заданных групп в обе стороны и проверить контроль продолжающегося движения.',acceptance=[dict(criterion=criterion,met=None)],done='',obtained='',interpretation='',decision='',cannot_claim='',metrics=[],evidence=[dict(kind='json',ref=str(OUT.relative_to(ROOT)/'protocol.json').replace('\\','/'),claim='Протокол до новых выходов')]))
    cat.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');journal('По прямому поручению автора зарегистрирована локализация предыстории. Шесть фиксированных групп состояния, двусторонний перенос, разворот и продолжение движения, независимые проверки. Никакого подбора групп по полученному качеству. Протокол: `runs/flyvis_partial_state_swap_20261106/protocol.json`.')
def run():
    import torch,cv2
    from tracking.flyvis_adapter import activate_flyvis
    from scripts.flyvis_y1_latency_fp64_20261104 import parameter_sha,cpu_eye
    from tracking.flyvis_state_transplant import transplant_nodes
    p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    for f,h in p['frozen_sources'].items():assert sha(ROOT/f)==h,f
    assert sha(OLD/'independent_audit.json')==p['old_audit_sha256'] and sha(SRC/'geometry.npz')==p['geometry_sha256']
    fv=activate_flyvis()
    for m in (fv,importlib.import_module('flyvis.network.initialization'),importlib.import_module('flyvis.task.decoder')):m.device=torch.device('cpu')
    torch.set_default_device('cpu');torch.set_default_dtype(torch.float64);torch.set_num_threads(2);cv2.setNumThreads(2)
    from tracking.flyvis_motion import FlyVisMotionEstimator
    model=FlyVisMotionEstimator(mode='full',source_fps=30,flow_gain=1,temporal_resampling='hold',subtract_initial_baseline=False,geometry_version='legacy_v1')
    model.network.double();model.decoder.double();model.eye.conv.double();net,dec,eye=model.network,model.decoder,model.eye
    net.eval().requires_grad_(False);dec.eval().requires_grad_(False);net.clamp();assert not net._state_hooks
    frozen=parameter_sha(net,dec);assert frozen==p['parameter_sha'], 'Model drift since full transplant'
    ix={g:np.concatenate([np.asarray(net.stimulus.layer_index[n]) for n in names]) for g,names in p['groups'].items()}
    partition=np.concatenate([ix[g] for g in GROUPS[:5]]);assert np.array_equal(np.sort(partition),np.arange(net.n_nodes))
    for v in ix.values():assert len(np.unique(v))==len(v) and v.min()>=0 and v.max()<net.n_nodes
    np.savez_compressed(OUT/'indices.npz',**ix)
    save(OUT/'model_loaded.json',dict(parameter_sha=frozen,n_nodes=net.n_nodes,group_nodes={g:len(v) for g,v in ix.items()},indices_sha256=sha(OUT/'indices.npz')))
    geo=np.load(SRC/'geometry.npz');assert np.array_equal(eye.receptor_centers.cpu().numpy(),geo['centers'])
    source_images=json.loads((SRC/'source_images.json').read_text())['sources'];textures={}
    for scene in sorted({c['scene'] for c in p['cases']}):
        info=source_images[scene];cap=cv2.VideoCapture(str(ROOT/'data/bdd100k/videos/train'/(info['video']+'.mov')));ok,bgr=cap.read();cap.release();assert ok
        gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY);assert hashlib.sha256(gray.tobytes()).hexdigest()==info['frame_sha256'];textures[scene]=gray
    np.savez_compressed(OUT/'textures.npz',**{str(k):v for k,v in textures.items()});started=time.monotonic()
    for ci,c in enumerate(p['cases']):
        assert datetime.now(ZoneInfo('Europe/Moscow'))<datetime.fromisoformat(p['deadline_msk']);assert shutil.disk_usage(OUT).free>=p['minimum_free_bytes']
        dest=OUT/'cases'/(c['id']+'.npz');marker=dest.with_suffix('.json')
        if marker.exists():
            m=json.loads(marker.read_text());assert sha(dest)==m['sha256'] and m['protocol_sha256']==sha(OUT/'protocol.json');continue
        begin=time.monotonic();refpath=OLD/'cases'/(c['id']+'.npz');inpath=SRC/'inputs'/(c['id']+'.npz')
        assert sha(refpath)==p['reference_sha256'][c['id']]['state_sha256'] and sha(inpath)==p['reference_sha256'][c['id']]['input_sha256']
        ref=np.load(refpath);boundary=torch.from_numpy(ref['boundary_activity']);assert boundary.shape==(2,net.n_nodes)
        a,b=boundary[0],boundary[1];before=boundary.clone()
        initial=[a.clone(),b.clone(),transplant_nodes(a,a,ix['t4t5']),transplant_nodes(b,b,ix['t4t5'])]
        for g in GROUPS:
            initial += [transplant_nodes(a,b,ix[g]),transplant_nodes(b,a,ix[g])]
        initial=torch.stack(initial);assert torch.equal(boundary,before)
        gray=textures[c['scene']];h,w=gray.shape;y0=h-512 if c['axis']==1 and c['speed']==4 else (h-512)//2;x0=(w-512)//2;frames=[]
        for t in range(24,49):
            shift=c['speed']*t;y=y0-(shift if c['axis']==1 else 0);x=x0-(shift if c['axis']==0 else 0)
            assert min(x,y)>=0 and x+512<=w and y+512<=h
            crop=gray[y:y+512,x:x+512];frames.append(np.rint(np.clip(127.5+c['contrast']*(crop.astype(np.float32)-127.5),0,255)).astype(np.uint8))
        frames=np.stack(frames);assert np.array_equal(frames[0],np.load(inpath)['frames'][54])
        with torch.inference_mode():
            cart=np.stack([cv2.resize(g,(391,391),interpolation=cv2.INTER_AREA) for g in frames]);rendered=eye(torch.from_numpy(cart.astype(np.float64)/255)[None])[0,:,0].cpu().numpy()
        eyediff=float(np.max(np.abs(rendered-cpu_eye(frames,geo['centers']))));assert eyediff<=1e-5
        assert np.max(np.abs(rendered[0]-ref['last_history_input']))<=1e-12
        flows=np.empty((2,len(ARMS),24,721,2),np.float64);state_identity=True;initial_saved=initial.cpu().numpy().copy()
        for scenario,suffix in enumerate([ref['suffix'],rendered[1:]]):
            with torch.inference_mode():
                state=net._initial_state(net._param_api(),len(ARMS));assert list(state.nodes)==['activity'] and not list(state.edges)
                state.nodes.activity=initial.clone();state=net._state_api(state)
                for j in range(24):
                    inp=torch.from_numpy(suffix[j])[None,None,None].expand(len(ARMS),2,1,721)
                    sequence=net.simulate(inp,model.dt,initial_state=state,as_states=True)
                    for s in sequence:
                        state_identity &= bool(torch.equal(s.nodes.activity[0],s.nodes.activity[2]) and torch.equal(s.nodes.activity[1],s.nodes.activity[3]))
                        assert torch.isfinite(s.nodes.activity).all()
                    state=sequence[-1];flows[scenario,:,j]=dec(state.nodes.activity[:,None])[:,0].cpu().numpy().transpose(0,2,1)*np.array([512/169,-512/169])
                    if j%6==0:save(OUT/'status.json',dict(stage='inference',done=ci,total=32,scenario=p['scenarios'][scenario],frame=j,case=c['id'],elapsed_seconds=time.monotonic()-started))
        refdiff=max(float(np.max(np.abs(flows[0,0]-ref['flow'][0]))),float(np.max(np.abs(flows[0,1]-ref['flow'][1]))))
        assert refdiff<=1e-10 and state_identity and torch.equal(boundary,before) and np.array_equal(initial.cpu().numpy(),initial_saved)
        assert np.isfinite(flows).all() and parameter_sha(net,dec)==frozen and not torch.cuda.is_initialized()
        dest.parent.mkdir(exist_ok=True);np.savez_compressed(dest,flow=flows,initial_states=initial_saved,continue_frames=frames,continue_hexes=rendered,reverse_hexes=ref['suffix'])
        save(marker,dict(sha256=sha(dest),protocol_sha256=sha(OUT/'protocol.json'),state_reference_sha256=sha(refpath),reference_diff=refdiff,eye_max_diff=eyediff,identity_states_exact=state_identity,donors_unchanged=True,parameters_and_buffers_unchanged=True,seconds=time.monotonic()-begin))
        if ci==0:
            required=2*dest.stat().st_size*31+5_000_000_000;assert shutil.disk_usage(OUT).free>=required
            save(OUT/'first_case_cost.json',dict(seconds=time.monotonic()-begin,remaining=31,forecast_seconds=31*(time.monotonic()-begin),disk_required=required,disk_passed=True))
        save(OUT/'status.json',dict(stage='case_complete',done=ci+1,total=32,elapsed_seconds=time.monotonic()-started))
    save(OUT/'inference_completed.json',dict(complete=True,cases=32,protocol_sha256=sha(OUT/'protocol.json'),parameter_sha=frozen,indices_sha256=sha(OUT/'indices.npz'),textures_sha256=sha(OUT/'textures.npz'),seconds=time.monotonic()-started,cuda_initialized=torch.cuda.is_initialized()))
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['register','run']);v=a.parse_args();register() if v.stage=='register' else run()
