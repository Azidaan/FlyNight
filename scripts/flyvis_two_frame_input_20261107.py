"""Fixed two-observed-input intervention after an identical retinal prefix/state."""
import os
import sys
import time
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from scripts.flyvis_input_separation_20261107 import read, sha, write, GUARD

BASE = Path('runs/flyvis_separate_inputs_20261107')
OUT = Path('runs/flyvis_two_frame_input_20261107')
IDENT = 'flyvis_two_frame_input_v1'

def prepare():
    assert not OUT.exists()
    OUT.mkdir()
    p = read(BASE / 'protocol.json')
    now = datetime.now(timezone(timedelta(hours=3)))
    p.update(id=IDENT, registered=now.isoformat(),
        git=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        author='Давай проверим раздельные вырезки на последних двух уже известных кадрах, сохранив одинаковую более раннюю память. Сначала убедимся, что сигнал действительно доходит до T4/T5.',
        hypothesis='Limiting the last TWO already observed retinal inputs to candidate windows, at the original scale and position, improves frozen target ranking relative to common and union inputs after an identical earlier state.',
        scope='Input preparation in the full FlyVis module; frozen core and readout. Historical exploratory pairs, not full tracker/HOTA or a new independent test.',
        intervention='Only inputs at T-2 and T-1 differ; prefix through T-3 and all recurrent state are identical. Current-box windows x1.25, original position/scale, neutral outside, same shared lamina, exact saved inputs. No centering, future frame, new integration step or sequence duplication.',
        controls=['common untouched', 'identity clone', 'union of two windows', 'separate A', 'separate B'],
        first_gate=dict(uid=p['cases'][0]['uid'],
            criterion='Before processing later cases: separated input must change final normalized T4 AND T5 maps by >1e-7, each exceeding identity difference. Each family may be exposed by either candidate. Otherwise STOP without changing K or the gate.',
            minimum_map_change=1e-7),
        parent_protocol_sha256=sha(BASE/'protocol.json'),
        parent_inputs_manifest_sha256=sha(BASE/'inputs_completed.json'),
        budget=dict(expected_gpu_s=180, watchdog_s=1200, system_memory='max(6GiB,20% physical) reserve; +2GiB for start; external supervisor', nearest_weekday_stop='2026-11-09T07:40:00+03:00', disk_reserve_gib=5))
    # Parent fingerprints retain exact original scientific sources, weights and inputs.
    p['source_sha256'][Path(__file__).relative_to(ROOT).as_posix()] = sha(Path(__file__))
    write(OUT/'protocol.json', p)
    for name in ['inputs', 'raw']:
        (OUT/name).mkdir()
    (OUT/'.gitignore').write_text('**/inputs/\n**/raw/\n*.png\n*.jpg\nvault_check_copy/\n', encoding='utf8')
    write(OUT/'status.json', dict(stage='registered', complete=0, total=172))
    cat = read('experiments/project_history.json')
    assert not any(x['id']==IDENT for x in cat['experiments'])
    cat['experiments'].append(dict(id=IDENT, title='Два последних входа FlyVis — отдельные окна при одинаковой ранней памяти', area='flyvis', state='ready',
        question='Доходит ли ограничение двух последних известных входов до T4/T5 и улучшает ли выбор между двумя рамками?',
        hypothesis=p['hypothesis'], expected='Прохождение сигнала до обоих семейств и положительные нижние границы обеих первичных разностей.',
        acceptance=[dict(criterion='Первая фиксированная пара: изменение T4 и T5 >1e-7 выше тождественного контроля; все172 выхода и численный независимый аудит с прежними допусками.',met=None),
                    dict(criterion='Separate-common и separate-union: обе нижние границы97,5% >0; 20k кластерных повторов, seed42, исходные26 видео.',met=None)],
        started=now.strftime('%Y-%m-%d'), summary='', obtained='', interpretation='', decision='',
        cannot_claim='Не HOTA; исторические исследовательские172 пары из23 видео,26 исходных кластеров. Ограниченные окна прежнего поля после общей lamina, не полные центрированные автомобили;129/344 рамок частично вне поля. Не доказывает устранение смешения объектов или пользу ядра в полном трекере.',
        evidence=[dict(kind='file',ref=(OUT/'protocol.json').as_posix(),claim='Условия и первая проверка прохождения сигнала до измерения новых выходов')]))
    write('experiments/project_history.json',cat)
    plan=Path('dissertation/ЕДИНЫЙ_ПЛАН.md')
    text=plan.read_text(encoding='utf8')
    entry=f'\n**{now:%d.%m.%Y, %H:%M} MSK** — Зарегистрирован `{IDENT}` по прямому поручению автора. Меняются только последние два уже доступных входа; более ранняя память одинакова. Условие допуска к172 парам: фиксированная первая пара должна изменить T4 и T5 выше1e-7 при тождественном контроле. Прежние входы переиспользуются по SHA; видео и BoxEye не пересчитываются. Протокол: `{OUT}/protocol.json`.\n'
    plan.write_text(text.replace('## Журнал изменений','## Журнал изменений\n'+entry,1),encoding='utf8')
    print('Registered', IDENT, flush=True)

def run(obs):
    import numpy as np
    import torch
    import shutil
    from scripts.z13_target_model import build_arm
    from tracking.gpu_lock import GpuLock
    p=read(OUT/'protocol.json'); ins=read(BASE/'inputs_completed.json')
    assert sha(BASE/'inputs_completed.json')==p['parent_inputs_manifest_sha256']
    for fn,h in p['source_sha256'].items():
        assert sha(fn)==h,fn
    for c in p['cases']:
        src=BASE/'inputs'/f"{c['uid']}.npz"; dst=OUT/'inputs'/src.name
        assert sha(src)==ins['sha256'][src.name]
        assert not dst.exists()
        os.link(src,dst)
    write(OUT/'inputs_completed.json',{**ins,'protocol_sha256':sha(OUT/'protocol.json')})
    norm=read('runs/z13_target_center_dev_20261018/C/normalization.json')
    mu=np.asarray(norm['mean'])[4:12];sd=np.asarray(norm['sd'])[4:12]
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    finished={}; exposure=[];start=time.perf_counter()
    with GpuLock('flyvis_fovea_gpu'),torch.no_grad():
        _,_,bb=build_arm('C',torch,None); params=bb.net._param_api()
        assert bb.steps==2
        for i,c in enumerate(p['cases']):
            assert shutil.disk_usage(ROOT).free>=5*1024**3
            uid=c['uid']; dest=OUT/'raw'/f'{uid}.npz'; before=time.perf_counter()
            assert not dest.exists()
            with np.load(OUT/'inputs'/dest.name,allow_pickle=False) as z:
                seq=z['inputs'].astype('float32');m=z['masks'];ref=z['reference'].astype('float64');w=z['weights']
            assert seq.ndim==3 and seq.shape[0]==4 and seq.shape[1]>=3 and seq.shape[2]==721
            hx=torch.from_numpy(seq[0])[None].to(bb.dev); state=bb.init(hx[:,:1])
            for j in range(1,hx.shape[1]-2):
                state,_=bb.step(state,hx[:,j-1],hx[:,j])
            assert list(state.nodes)==['activity'] and not list(state.edges)
            original=state.nodes.activity.clone()
            cl=bb.net._initial_state(params,5);cl.nodes.activity=original.expand(5,-1).clone();cl=bb.net._state_api(cl)
            assert torch.equal(cl.nodes.activity,original.expand(5,-1))
            previous=hx[:,-3].expand(5,-1)
            for j in [-2,-1]:
                cur=torch.from_numpy(np.stack([seq[0,j],seq[0,j].copy(),seq[1,j],seq[2,j],seq[3,j]])).to(bb.dev)
                cl,f=bb.step(cl,previous,cur); previous=cur
            assert torch.equal(state.nodes.activity,original)
            maps=((f[:,4:12].cpu().numpy().astype('float64')-mu[None,:,None])/sd[None,:,None]).astype('float32')
            scores=np.stack([maps[:,:,mask].astype('float64').mean(2)@w for mask in m],1)
            refs=np.array([ref[:,mask].mean(1)@w for mask in m])
            am=float(np.abs(maps[0]-ref).max());im=float(np.abs(maps[1]-maps[0]).max())
            assert np.isfinite(maps).all() and am<=p['gates']['archive_normalized_map_max']
            assert np.abs(scores[0]-refs).max()<=p['gates']['archive_score_max']
            rank=lambda x:int(x>1e-10)-int(x< -1e-10)
            assert rank(scores[0,0]-scores[0,1])==rank(refs[0]-refs[1])
            assert im<=p['gates']['identity_map_max'] and np.max(np.abs(scores[1]-scores[0]))<=p['gates']['identity_score_max']
            with dest.open('xb') as stream:
                np.savez_compressed(stream,uid=uid,maps=maps,masks=m,weights=w,reference=ref,archive_map_max=am,identity_map_max=im,worker_scores=scores)
            finished[dest.name]=sha(dest);torch.cuda.synchronize()
            effect=dict(uid=uid,t4_max=float(np.abs(maps[3:,:4]-maps[0,:4]).max()),t5_max=float(np.abs(maps[3:,4:]-maps[0,4:]).max()),identity_max=im,
                changed_last_inputs=int(np.count_nonzero(seq[2:,-2:]!=seq[0,-2:])))
            exposure.append(effect)
            if i==0:
                floor=p['first_gate']['minimum_map_change']
                passed=all(effect[k]>max(floor,im) for k in ['t4_max','t5_max'])
                write(OUT/'first_case.json',dict(**effect,passed=passed,seconds=time.perf_counter()-before,remaining_estimate_s=(time.perf_counter()-before)*171))
                assert passed,'First fixed case did not expose BOTH T4 and T5; stop, no quality inference.'
            write(OUT/'status.json',dict(stage='two_frame_gpu',complete=i+1,total=172,seconds=time.perf_counter()-start,guard=obs.read()))
    for fn,h in p['source_sha256'].items():
        assert sha(fn)==h,fn
    write(OUT/'exposure.json',dict(cases=exposure))
    write(OUT/'inference_completed.json',dict(complete=True,count=172,raw_sha256=finished,seconds=time.perf_counter()-start,guard=obs.read(),protocol_sha256=sha(OUT/'protocol.json')))

def audit():
    from scripts import audit_flyvis_separate_inputs_20261107 as evaluator
    evaluator.OUT=OUT
    evaluator.audit()
    import numpy as np
    rows=[]
    p=read(OUT/'protocol.json')
    for c in p['cases']:
        with np.load(OUT/'raw'/f"{c['uid']}.npz",allow_pickle=False) as z:
            x=z['maps'].astype('float64')
            rows.append(dict(uid=c['uid'], t4_max=float(abs(x[3:,:4]-x[0,:4]).max()),t5_max=float(abs(x[3:,4:]-x[0,4:]).max()),identity_max=float(abs(x[1]-x[0]).max())))
    first=read(OUT/'first_case.json'); assert first['uid']==rows[0]['uid']
    for k in ['t4_max','t5_max','identity_max']:
        assert first[k]==rows[0][k],k
    assert first['passed'] and all(first[k]>max(first['identity_max'],p['first_gate']['minimum_map_change']) for k in ['t4_max','t5_max'])
    write(OUT/'exposure_audit.json',dict(passed=True,count=len(rows),rows=rows,t4_changed=sum(r['t4_max']>1e-7 for r in rows),t5_changed=sum(r['t5_max']>1e-7 for r in rows),first_gate_exact=True))

if __name__=='__main__':
    if '--prepare' in sys.argv:
        prepare()
    elif '--audit' in sys.argv:
        audit()
    else:
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
        sys.path.insert(0,str(GUARD))
        from system_memory_guard import guard_entrypoint
        run(guard_entrypoint(timeout=1200))
