"""Matched CPU/GPU strata for readout training; same data, FP32, seeds and deadlines."""
from pathlib import Path
import os,sys,inspect,copy,hashlib,time
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts import z13_spatial_set_recovery3_20261025 as prior
RUN=prior.RUN;PREV=prior.SERVICE;SERVICE=RUN/'training/recovery4_gpu'


def accelerated_predict(torch,model,data,idx,points):
    out=[];model.eval();device=next(model.parameters()).device;points=points.to(device)
    with torch.no_grad():
        for left in range(0,len(idx),64):
            b={k:v.to(device) for k,v in take(torch,data,idx[left:left+64]).items()}
            z=model(b['x'],b['seed'],b['geo'],b['mask'],b['full'],b['retinal_mask'],b['centers'],b['scales'],points)
            out.extend(v[:int(m.sum())].numpy().copy() for v,m in zip(z.cpu(),b['mask'].cpu()))
    return out


def source():
    s=prior.source();a=s.index('def predict(');b=s.index('\n\ndef one(',a)
    s=s[:a]+inspect.getsource(accelerated_predict).replace('def accelerated_predict(','def predict(',1)+s[b:]
    prefix="def train_worker():\n    os.environ['CUDA_VISIBLE_DEVICES']=''\n    import torch\n    torch.set_num_threads(2);torch.set_default_device('cpu');p=read(RUN/'training/protocol.json');verify_sources(p)"
    config="def train_worker():\n    os.environ['CUDA_VISIBLE_DEVICES']='0'\n    import torch\n    torch.set_num_threads(2);torch.set_default_device('cpu');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)\n    assert torch.cuda.is_available() and torch.cuda.device_count()==1\n    _probe=torch.ones(1,device='cuda:0');assert float(_probe.sum())==1.;del _probe\n    p=read(RUN/'training/protocol.json');verify_sources(p)"
    changes=[(prefix,config),
      ("folder.mkdir(parents=True,exist_ok=True);vi=data['video'].numpy();kind=data['kind'].numpy()","device='cpu' if seed==p['seeds'][0] and split['validation']==p['folds'][0]['validation'] else 'cuda:0'\n    points=points.to(device)\n    folder.mkdir(parents=True,exist_ok=True);vi=data['video'].numpy();kind=data['kind'].numpy()"),
      ("activity=arm!='geometry');opt=torch.optim.Adam(model.parameters(),lr=p['learning_rate'])","activity=arm!='geometry').to(device);opt=torch.optim.Adam(model.parameters(),lr=p['learning_rate'])"),
      ("b=take(torch,data,rng.choice(train,p['batch'],replace=False));model.train()","b=take(torch,data,rng.choice(train,p['batch'],replace=False));b={k:v.to(device) for k,v in b.items()};model.train()"),
      ('replay=predict(torch,restored,data,val,points)','restored.to(device);replay=predict(torch,restored,data,val,points)'),
      ('if models==1:', 'if models==2:'),
      ("estimate=2*r['seconds']*89+600;left=","estimate=2*r['seconds']*84+2*CPU_FIRST_SECONDS*4+600;left="),
      ("write(RUN/'training/price.json',dict(estimate=estimate,remaining=left,proceed=estimate<left))","write(ACCEL_SERVICE/'price.json',dict(estimate=estimate,remaining=left,proceed=estimate<left,remaining_gpu_models=84,remaining_cpu_models=4,cpu_reference_seconds=CPU_FIRST_SECONDS))"),
      ("data=load_data(torch,arm,entries)\n", "data=load_data(torch,arm,entries);os.environ['CUDA_VISIBLE_DEVICES']='0'\n"),
      ("data=load_data(torch,arm,entries,include26=True)\n", "data=load_data(torch,arm,entries,include26=True);os.environ['CUDA_VISIBLE_DEVICES']='0'\n")]
    for a,b in changes:assert s.count(a)==1,(a,s.count(a));s=s.replace(a,b)
    return s


def device_check():
    import numpy as np
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    import torch
    torch.set_num_threads(2);torch.set_default_device('cpu');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    from tracking.z13_spatial_set_readout import make_model
    from tracking.z13_default_class import action_loss
    old=ROOT/'runs/z13_region_assignment_20261025/recovery2/features/D/fit20/chunk_00000.npz'
    with np.load(old) as z:original={k:z[k].copy() for k in z.files}
    path=RUN/'features/D/fit20/chunk_00000.npz';assert sha(path)==read(path.with_suffix('.json'))['sha256']
    with np.load(path) as z:d={k:z[k][:4].copy() for k in z.files}
    lookup={int(u):i for i,u in enumerate(original['uid'])};ix=[lookup[int(u)] for u in d['uid']]
    geo=torch.cat([torch.tensor(original['geo'][ix]),torch.zeros((4,d['pooled'].shape[1],6))],-1)
    args=[torch.tensor(d['pooled']),torch.tensor(original['seed'][ix]),geo,torch.tensor(d['retinal_mask'].any(-1)),torch.tensor(d['full']),torch.tensor(d['retinal_mask']),torch.tensor(d['centers']),torch.tensor(d['scales']),torch.tensor(read(ROOT/'runs/z13_local_peak_20261021/geometry.json')['points'],dtype=torch.float32)]
    weights=RUN/'training/internal/D_spatial/seed0/fold0/model.pt';done=read(weights.with_name('done.json'));assert sha(weights)==done['model_sha256']
    cpu=make_model(torch);cpu.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True));gpu=make_model(torch);gpu.load_state_dict(cpu.state_dict());gpu.to('cuda:0')
    good=torch.zeros_like(args[3]);harm=~good;harm[:,0]=True;good[:,1]=True;harm[:,1]=False
    zcpu=cpu(*args);lc=action_loss(torch,zcpu,good,harm,args[3],'cost_ce',.01);lc.backward()
    ga=[a.to('cuda:0') for a in args];zgpu=gpu(*ga);lg=action_loss(torch,zgpu,good.to('cuda:0'),harm.to('cuda:0'),ga[3],'cost_ce',.01);lg.backward()
    delta=float((zcpu.detach()-zgpu.detach().cpu()).abs().max());gradient=max(float((a.grad-b.grad.cpu()).abs().max()) for a,b in zip(cpu.parameters(),gpu.parameters()) if a.grad is not None)
    assert delta<1e-5 and gradient<1e-5,(delta,gradient)
    restored=make_model(torch);restored.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True));restored.to('cuda:0')
    with torch.no_grad():replayed=restored(*ga)
    assert torch.equal(zgpu.detach(),replayed)
    write(SERVICE/'backend_check.json',dict(at=now().isoformat(),device='cuda:0',dtype='float32',tf32=False,deterministic_algorithms=True,logit_max_difference_cpu_gpu=delta,gradient_max_difference_cpu_gpu=gradient,new_backend_check_tolerance=1e-5,saved_weight_gpu_replay_exact=True,passed=True,weight_sha256=sha(weights),scope='Numerical backend check only; no model fitting, outcome evaluation or changed scientific acceptance.'))


def prepare():
    from scripts.z13_hota_deadline_guard import WindowsProcesses
    from scripts.z13_localization_labels_20261023 import verify_sources
    assert not (SERVICE/'protocol.json').exists();p=read(PREV/'protocol.json');verify_sources(p);ps=WindowsProcesses()
    e=read(PREV/'controller_result.json');assert e['exit_code']==1 and e['all_owned_ended'] and not e['remaining']
    assert all(ps.creation(int(pid))!=b for pid,b in e['owned'].items()) and not read(RUN/'training/price.json')['proceed']
    done_files=list((RUN/'training/internal').glob('*/seed*/fold*/done.json'));assert len(done_files)==1
    first=RUN/'training/internal/D_spatial/seed0/fold0';done=read(first/'done.json');assert sha(first/'model.pt')==done['model_sha256'] and sha(first/'report.json')==done['report_sha256'] and done['predictions_reproduced_exact']
    assert read(SERVICE/'backend_check.json')['passed'];q=copy.deepcopy(p)
    q.update(device='cuda',stop_request_path=(SERVICE/'stop.request').relative_to(ROOT).as_posix(),executed_service_sha256=hashlib.sha256(source().encode()).hexdigest(),gpu_backend=dict(at=now().isoformat(),reason='CPU actual first-model conservative estimate14019.60s exceeded remaining9949.20s; preserve gate failure and completed model. User explicitly requested GPU25Oct.',schedule='seed0/outerfold0 onCPU for EVERYarm, reusing existing completed D model; other14seed/fold blocks onGPU for EVERYarm. Any conditional final15models allGPU. Schedule fixed from cost, before evaluating comparison outcomes.',precision='FP32;TF32disabled;deterministic algorithms enabled. Models initialized onCPU using originaltorch.manual_seed then moved to backend; same minibatches/seeds/loss/optimizer/steps.',reused_model=first.relative_to(ROOT).as_posix(),reused_model_sha256=done['model_sha256'],cpu_reference_seconds=read(first/'report.json')['seconds'],original_cpu_save_at=p['save_at'],original_cpu_hard_stop=p['hard_stop'],criterion='Same all5primary comparisons and2pp criterion. Exact saved prediction replay within each backend. New mathematical backend check1e-5 does NOT replace any old reproduction/center/score tolerance.',price='First NEW GPUmodel:2*measured_seconds*84remainingGPU+2*CPUreference_seconds*4remainingCPU+600 < remaining original deadline. Formula retains2xconservative factor and600overhead; completed CPUmodel never repeated.'))
    names=[Path(__file__).relative_to(ROOT).as_posix(),(SERVICE/'backend_check.json').relative_to(ROOT).as_posix(),(PREV/'controller_result.json').relative_to(ROOT).as_posix(),(RUN/'training/price.json').relative_to(ROOT).as_posix(),(first/'done.json').relative_to(ROOT).as_posix(),(first/'report.json').relative_to(ROOT).as_posix()]
    q['sha256']={**p['sha256'],**{n:sha(ROOT/n) for n in names}};write(SERVICE/'protocol.json',q);write(SERVICE/'status.json',dict(state='ready',stage='Одинаковые CPU/GPU блоки для всех вариантов',updated=now().isoformat()))
    ns=namespace(verify=False)
    ns['journal']('перенос считывателей на GPU с равными вычислительными блоками','*Получено.* RAM-сборка завершилась и перваяCPUмодель D_spatial/seed0/fold0 обучена1150шагов; веса/предсказания точно воспроизведены,75,391с. Прежняя CPU-оценка14019,598с>9949,204с оставшегося времени, gatefalse сохранён; это ограничение стоимости, не отрицательный научный итог.\n\n*Новый порядок исполнения до продолжения.* По ранее прямой просьбе автора «Давай гпу» вычислительный блок seed0/fold0 остаётсяCPU для всех5вариантов, перваяD переиспользуется. Остальные14блоков всехвариантов —GPU; условные15итоговых всеGPU. Разбиения/сид/шаги/данные/цели/критерии не меняются, лучшийрезультат не выбирается. Сохраняются FP32, выключенTF32, включены детерминированные алгоритмы; backend_check.json проверил логиты/градиенты и точный повтор из весов наGPU. Это новый аппаратный контроль, прежние допуски не ослаблены.\n\n*Бюджет.* Новый GPU-controller ограничен тем же '+p['save_at']+' / '+p['hard_stop']+'; CPUпредел не продлён. Прежние исходники/коды/цена/готоваямодель сохранены. Источник: training/recovery4_gpu/protocol.json.')
    cat=read(ROOT/'experiments/project_history.json');card=next(x for x in cat['experiments'] if x['id']==p['id']);card['evidence'].append(dict(kind='file',ref=(SERVICE/'protocol.json').relative_to(ROOT).as_posix(),claim='Равное аппаратное расписание:CPUseed0fold0 для всех,остальныеGPU; перваяCPUмодель сохранена, критерии и срок прежние'));write(ROOT/'experiments/project_history.json',cat)
    ns['handoff']('ТЕКУЩИЙ spatial-set training/recovery4_gpu: '+SERVICE.relative_to(ROOT).as_posix()+'. Все1863GPUблока готовы,неповторять. RAMrecovery3 устранилflush,полностьюготова1CPUмодель D_spatial/seed0/fold0,точныйreplay;CPUprice14019,60>9949,20остановил. Теперьseed0/fold0CPU ДЛЯВСЕХ5вариантов,остальные14блоковGPUДЛЯВСЕХ; перваяDнеповторяется. Backend FP32/TF32off/deterministic проверен. ОдноGPUдерево,GpuLock,скрытыйIdle; прежнийsave '+p['save_at']+' hard '+p['hard_stop']+'. Статус/модели/report в recovery2/training, controller/result/backendprice в recovery4_gpu. Старыеbase/root/recovery1/recovery3ошибки не являются текущим статусом. Сначала новыйpricegate, затем75моделей, условно15итоговых по всем5ДИи2п.п. НикакогоHOTA. Послеитога аудитвсехстрок/хешей/карточка/revisions/журнал/тесты/preserving graph/коммит/push.57чужихправок/.gitignoreсохранить.')


def namespace(verify=True):
    s=source();first=RUN/'training/internal/D_spatial/seed0/fold0/report.json'
    if verify:assert hashlib.sha256(s.encode()).hexdigest()==read(SERVICE/'protocol.json')['executed_service_sha256']
    ns={'__name__':'spatial_set_gpu_readout_recovery','__file__':str(ROOT/'scripts/z13_spatial_set_readout_20261025.py'),'available_memory':prior.available_memory,'ACCEL_SERVICE':SERVICE,'CPU_FIRST_SECONDS':read(first)['seconds']};exec(compile(s,__file__,'exec'),ns);return ns


if __name__=='__main__':
    if '--device-check' in sys.argv:device_check()
    elif '--prepare' in sys.argv:prepare()
    elif '--worker' in sys.argv:
        os.environ['CUDA_VISIBLE_DEVICES']='0';ns=namespace()
        from tracking.gpu_lock import GpuLock
        try:
            with GpuLock('flyvis_fovea_gpu'):ns['train_worker']()
            write(SERVICE/'status.json',dict(state='completed',stage='Серия сохранена',updated=now().isoformat()))
        except Exception as exc:
            write(SERVICE/'failure.json',dict(at=now().isoformat(),error=repr(exc)));write(SERVICE/'status.json',dict(state='failed',stage=repr(exc),updated=now().isoformat()));raise
    else:raise RuntimeError('Explicit mode required')
