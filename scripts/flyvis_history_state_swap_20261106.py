"""Prospective full recurrent-state transplant with identical continuation."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse, hashlib, importlib, json, sys, time, shutil
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'runs/flyvis_history_state_swap_20261106'
SRC = ROOT/'runs/flyvis_offset_recovery2_20261104'
sys.path.insert(0, str(ROOT))
ARMS = ['moving', 'stationary', 'identity_moving', 'moving_gets_stationary', 'stationary_gets_moving', 'identity_stationary']
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_suffix('.next')
    q.write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    for i in range(40):
        try: os.replace(q, p); return
        except PermissionError:
            if i == 39: raise
            time.sleep(.25)
def journal(text):
    path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';raw=path.read_bytes();marker='## Журнал изменений'.encode()
    nl=b'\r\n' if marker+b'\r\n' in raw else b'\n';assert raw.count(marker+nl)==1
    stamp=datetime.now(ZoneInfo('Europe/Moscow')).strftime('%d.%m.%Y, %H:%M')
    path.write_bytes(raw.replace(marker+nl,marker+nl+f'\n**{stamp} MSK** — Подмена полного состояния FlyVis\n\n{text}\n\n'.encode(),1))
def register():
    OUT.mkdir(exist_ok=False)
    prior=json.loads((SRC/'protocol.json').read_text(encoding='utf-8'))
    cases=[c for c in prior['cases'] if c['start']=='still' and c['split']=='evaluation']
    assert len(cases)==32 and len({c['scene'] for c in cases})==4
    sources=['tracking/flyvis_motion.py','tracking/flyvis_adapter.py','tools/flyvis-runtime/flyvis/network/network.py','tools/flyvis-runtime/flyvis/network/dynamics.py','tools/flyvis-runtime/flyvis/task/decoder.py','scripts/flyvis_y1_latency_fp64_20261104.py','runs/flyvis/results/flow/0000/000/chkpts/chkpt_00000', 'scripts/flyvis_history_state_swap_20261106.py','scripts/audit_flyvis_history_state_swap_20261106.py','scripts/control_flyvis_history_state_swap_20261106.py']
    refs={c['id']:sha(SRC/'stock/outputs'/(c['id']+'.npz')) for c in cases}
    criterion='All 32 cases: identity state copies exact; full transplant matches donor continuation states and decoded fields <=1e-10; native moving matches previous stock hold/last <=1e-5; finite outputs and no mutation of weights/donor. Influence is present where native history output max difference >1e-5; no improvement acceptance claim.'
    p=dict(id='flyvis_history_state_swap_v1',registered_msk=datetime.now(ZoneInfo('Europe/Moscow')).isoformat(),deadline_msk='2026-11-07T01:00:00+03:00',cases=cases,arms=ARMS,source_protocol_sha256=sha(SRC/'protocol.json'),reference_sha256=refs,frozen_sources={s:sha(ROOT/s) for s in sources},criterion=criterion,
      method='Stock flow/0000/000 CPU FP64. A replays saved frames0..54 with 1s fade-in at frame0. B fades in frame54 for1s and holds frame54 for54 video steps. Thus both end on exactly the same frame54, but histories differ. At boundary AFTER frame54 and BEFORE frame55 copy ALL nodes/edges dynamic variables, rebuilding source/target references. Common suffix frames55..78, hold two integration steps/frame, last-step decoder, no interpolation or temporal averaging. Both directions of full state transplant plus independent cloned identity controls. Decoder eval with fixed buffers. Stationary donor is constructed diagnostic, not deployable reset.',
      metrics='No calibration or baseline subtraction. Central-mask mean nominal pixel/frame vectors: EPE over24 suffix frames, first3 consecutive correct negative signs (missing explicit and capped24), correct-sign fraction, mean vector trajectory difference. Same old coordinate convention gain1, x*512/169,y*-512/169. Report EPE and latency jointly, both contrasts; descriptive paired scene bootstrap 20k seed42 95% CI over4 scene means. These four evaluation images were studied previously; no claim of new heldout test.',
      decision='If full transplant does not reproduce donor, stop interpretation and debug missing state. If A/B differ and transplants reproduce donor, history effect is carried by complete recurrent state under this fixed suffix; does not localize a cell group, prove pathological memory, or solve delay. Next partial interventions require separate justified design.',
      limitations='One model;4 previously studied BDD TRAIN images x2 contrasts x2 speeds x2 axes; artificial whole-scene motion, not natural video or HOTA. No fresh dataset, training, GPU or altered core parameters. Difference can include normal adaptation to history, not solely harmful inertia.',watchdog_seconds=600,minimum_free_bytes=5_000_000_000)
    save(OUT/'protocol.json',p);save(OUT/'status.json',dict(stage='registered',done=0,total=32))
    cat=ROOT/'experiments/project_history.json';d=json.loads(cat.read_text(encoding='utf-8'));assert not any(c['id']==p['id'] for c in d['experiments'])
    d['experiments'].append(dict(id=p['id'],title='FlyVis — влияние предыстории при полной подмене состояния',area='flyvis',state='planned',started='2026-11-06',tags=['Y1','motion-diagnostic'],question='Переносит ли полное состояние ядра влияние предыстории на одинаковое продолжение?',hypothesis='Разные предыстории могут давать разные ответы; при полной подмене состояния продолжение воспроизведёт донора.',expected='Разделить влияние предыстории и исправность переноса без заявления об улучшении.',acceptance=[dict(criterion=criterion,met=None)],done='',obtained='',interpretation='',decision='',cannot_claim='',metrics=[],evidence=[dict(kind='json',ref=str(OUT.relative_to(ROOT)/'protocol.json'),claim='Условия до вычисления новых выходов')]))
    cat.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    journal('По прямому поручению автора зарегистрирована проверка полного состояния ядра при одинаковом продолжении. Две предыстории, двусторонняя подмена и тождественные контроли; 32 условия на четырёх ранее изученных изображениях. Настройки, критерии и ограничения: `runs/flyvis_history_state_swap_20261106/protocol.json`. До результатов никаких выводов об устранении задержки нет.')
def run():
    import torch
    from tracking.flyvis_adapter import activate_flyvis
    from scripts.flyvis_y1_latency_fp64_20261104 import parameter_sha
    p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    for f,h in p['frozen_sources'].items(): assert sha(ROOT/f)==h, f
    fv=activate_flyvis()
    for m in (fv,importlib.import_module('flyvis.network.initialization'),importlib.import_module('flyvis.task.decoder')):m.device=torch.device('cpu')
    torch.set_default_device('cpu');torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    from tracking.flyvis_motion import FlyVisMotionEstimator
    model=FlyVisMotionEstimator(mode='full',source_fps=30,flow_gain=1,temporal_resampling='hold',subtract_initial_baseline=False,geometry_version='legacy_v1')
    model.network.double();model.decoder.double();model.eye.conv.double()
    net,dec=model.network,model.decoder;net.eval().requires_grad_(False);dec.eval().requires_grad_(False);net.clamp()
    assert not net._state_hooks
    frozen=parameter_sha(net,dec)
    def buffers():return {k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k,v in dec.named_buffers()}
    initial_buffers=buffers();started=time.monotonic()
    def compose(states, indices):
        cls=type(states[0]);parts={}
        for section in ('nodes','edges'):
            keys=list(states[0][section]);assert all(list(s[section])==keys for s in states)
            parts[section]=cls(**{k:torch.cat([states[s][section][k][i:i+1].detach().clone() for s,i in indices],0) for k in keys})
        return net._state_api(cls(**parts))
    def states_numpy(s):
        return {f'{section}_{k}':s[section][k].detach().cpu().numpy().copy() for section in ('nodes','edges') for k in s[section]}
    for ci,c in enumerate(p['cases']):
        assert datetime.now(ZoneInfo('Europe/Moscow'))<datetime.fromisoformat(p['deadline_msk'])
        assert shutil.disk_usage(OUT).free>=p['minimum_free_bytes']
        dest=OUT/'cases'/(c['id']+'.npz');marker=dest.with_suffix('.json')
        if marker.exists():
            m=json.loads(marker.read_text());assert m['sha256']==sha(dest) and m['protocol_sha256']==sha(OUT/'protocol.json');continue
        q=SRC/'stock/outputs'/(c['id']+'.npz');assert sha(q)==p['reference_sha256'][c['id']]
        ref=np.load(q);hexes=ref['hexes'];flow=np.empty((6,24,721,2));state_diffs=[];step_hashes=[];begin=time.monotonic()
        with torch.inference_mode():
            # Separate fade-in for each history, equal duration and same checkpoint.
            first=torch.from_numpy(np.stack([hexes[0],hexes[54]]))[:,None]
            history=net.fade_in_state(1.,model.dt,first)
            for j in range(1,55):
                x=torch.from_numpy(np.stack([hexes[j],hexes[54]]))[:,None,None].expand(2,2,1,721)
                history=net.simulate(x,model.dt,initial_state=history,as_states=True)[-1]
                if j%15==0:save(OUT/'status.json',dict(stage='history',done=ci,total=32,frame=j,case=c['id'],elapsed_seconds=time.monotonic()-started))
            assert list(history.nodes)==['activity'] and not list(history.edges), 'Unexpected extra dynamical variables: review full-state definition'
            boundary=states_numpy(history);before=boundary['nodes_activity'].copy()
            # A,B,Aclone,A<-B,B<-A,Bclone. All source/target aliases rebuilt.
            state=compose([history],[(0,i) for i in [0,1,0,1,0,1]])
            assert state.nodes.activity.data_ptr()!=history.nodes.activity.data_ptr()
            for j in range(55,79):
                x=torch.from_numpy(hexes[j])[None,None,None].expand(6,2,1,721)
                seq=net.simulate(x,model.dt,initial_state=state,as_states=True)
                for s in seq:
                    vals=states_numpy(s);diffs=[]
                    for v in vals.values():
                        assert np.isfinite(v).all()
                        diffs.extend(float(np.max(np.abs(v[a]-v[b]))) for a,b in [(0,2),(1,3),(0,4),(1,5)])
                    state_diffs.append(diffs)
                    step_hashes.append([hashlib.sha256(s.nodes.activity[k].cpu().numpy().tobytes()).hexdigest() for k in range(6)])
                state=seq[-1]
                flow[:,j-55]=dec(state.nodes.activity[:,None])[:,0].cpu().numpy().transpose(0,2,1)*np.array([512/169,-512/169])
            assert np.array_equal(before,history.nodes.activity.cpu().numpy()), 'Donor mutated'
        native_diff=float(np.max(np.abs(flow[0]-ref['flow'][0,1,55:79])))
        assert native_diff<=1e-5, native_diff
        assert np.isfinite(flow).all() and np.max(state_diffs)<=1e-10
        assert parameter_sha(net,dec)==frozen and buffers()==initial_buffers and not torch.cuda.is_initialized()
        dest.parent.mkdir(exist_ok=True)
        np.savez_compressed(dest,flow=flow,boundary_activity=before,suffix=hexes[55:79],last_history_input=hexes[54],state_diffs=np.asarray(state_diffs))
        save(marker,dict(sha256=sha(dest),protocol_sha256=sha(OUT/'protocol.json'),reference_sha256=sha(q),native_diff=native_diff,step_state_sha256=step_hashes,donor_unchanged=True,parameters_unchanged=True,decoder_buffers_unchanged=True,state_schema=dict(nodes=['activity'],edges=[],sources_targets='derived references rebuilt'),seconds=time.monotonic()-begin))
        if ci==0:save(OUT/'first_case_cost.json',dict(seconds=time.monotonic()-begin,remaining=31,forecast_seconds=31*(time.monotonic()-begin)))
        save(OUT/'status.json',dict(stage='inference',done=ci+1,total=32,elapsed_seconds=time.monotonic()-started))
    save(OUT/'inference_completed.json',dict(complete=True,cases=32,parameter_sha=frozen,decoder_buffer_sha=initial_buffers,protocol_sha256=sha(OUT/'protocol.json'),seconds=time.monotonic()-started,cuda_initialized=torch.cuda.is_initialized()))
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['register','run']);v=a.parse_args();register() if v.stage=='register' else run()
