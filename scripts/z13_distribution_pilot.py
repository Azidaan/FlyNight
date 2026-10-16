"""Bounded matched continuation: one-hot versus spatial targets, frozen B/C/D cores."""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
SRC=ROOT/'runs/z13_spatial_v2_20261013'
OUT=ROOT/'runs/z13_distribution_pilot_20261015'
DEADLINE='2026-10-15T20:00:00+03:00'
SAVE_DEADLINE='2026-10-15T19:55:00+03:00'

def read(path): return json.loads(path.read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def now(): return datetime.now(timezone(timedelta(hours=3)))
def write(path,obj):
    from tracking.z13_checkpoint import atomic_json
    atomic_json(path,obj)

def prepare():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'protocol.json').exists()
    original=read(SRC/'protocol.json')
    sources=[ROOT/'tracking/z13_distribution_target.py',Path(__file__),
             ROOT/'scripts/z13_spatial_train.py', ROOT/'scripts/z13_train.py',
             ROOT/'tracking/z13_readout.py',ROOT/'tracking/z13_spatial.py',
             ROOT/'tracking/z13_checkpoint.py',SRC/'protocol.json']
    for a in 'BCD': sources.extend([SRC/a/'full/model.pt',SRC/a/'full/identity.json'])
    for v in sorted(set(original['fit']+original['tau'])): sources.append(ROOT/'runs/z13_data'/f'{v}.npz')
    p=dict(id='z13_distribution_pilot_v1',created=now().isoformat(),
        authorization='Author 15.10: if tree idea not useful continue the plan autonomously until 20:00 MSK',
        question='Does a spatial target improve localization compared with identical extra training on a one-hot target?',
        scope='One functional pilot on all existing train104/26; no dev, HOTA, downloads or heldout',
        fit=original['fit'],evaluation=original['tau'],arms=list('BCD'),losses=['hard','spatial'],
        steps=500,batch=16,micro=4,seed=0,lr=.003,sigma='one eye-column spacing',
        initialization='Each pair starts from the same completed arm full/model.pt; fresh identical Adam, identical sampled rows',
        trainable='Readout only; all B/C/D feature-extractor weights frozen, including already-trained D',
        labels='Positive normalized Gaussian at original continuous target center; known absence one-hot; unknown excluded',
        inference='Same center_estimate and output API; no quality scorer refit or tracker intervention',
        acceptance=['D spatial minus D hard useful-position fraction on known-positive rows: paired video CI lower >0',
                    'D spatial mean of per-video median relative center errors <= D hard',
                    'Separate FlyVis signal gate: D spatial minus B spatial useful-position fraction CI lower >0'],
        stop='Failure closes this fixed-scale readout-loss pilot; no sigma/lr sweep or automatic full tracker run',
        limits=dict(save_deadline=SAVE_DEADLINE,hard_deadline=DEADLINE,one_gpu_worker=True),
        caveats=['500-step continuation is a pilot, not convergence or HOTA evidence',
                 'Calibration videos previously used; exploratory intervals only',
                 'Fixed width is a grid-scale design, not measured annotation noise',
                 'Frozen trained D does not test further end-to-end core adaptation'],
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources})
    write(OUT/'protocol.json',p)
    cpath=ROOT/'experiments/project_history.json'; cat=read(cpath)
    assert not any(c['id']==p['id'] for c in cat['experiments'])
    cat['experiments'].append(dict(id=p['id'],title='З13: распределённая пространственная цель считывания, ограниченный пилот',
        area='flyvis',state='planned',started='2026-10-15',question=p['question'],
        hypothesis='Учёт расстояния до цели в обучении считывания может уменьшить ошибку положения; причинность прежнего провала не установлена.',
        expected='Парные продолжения hard/spatial из одинаковых весов B/C/D, 500 шагов каждое, только считыватель; train104/26. Одно окно до 20:00 MSK.',
        acceptance=[dict(criterion=x,met=None) for x in p['acceptance']],
        cannot_claim='Не проверяются HOTA, независимое подтверждение, новое обучение ядра или реализация всего PrDiMP.',
        evidence=[dict(kind='json',ref='runs/z13_distribution_pilot_20261015/protocol.json',claim='Постановка до обучения'),
                  dict(kind='url',ref='https://arxiv.org/abs/2003.12565',claim='Источник принципа пространственной цели; не готовая FlyVis')]))
    write(cpath,cat)
    path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md'; text=path.read_text(encoding='utf-8')
    marker='## Журнал изменений\n\n'; assert text.count(marker)==1
    entry=f"**{now():%d.%m.%Y, %H:%M} MSK** — подготовлен парный пилот пространственной цели считывания\n\n*Постановка (`z13_distribution_pilot_v1`).* После отрицательной пробы деревьев, по разрешению автора продолжать самостоятельно до 20:00, выбран следующий кандидат прежнего обзора `research_z13_spatial_readout_2026-10-13/СВОДКА.md`: принцип распределённой цели PrDiMP (https://arxiv.org/abs/2003.12565), адаптированный к решётке глаза. Это не перенос всего PrDiMP. Меняется только функция обучения считывателя: ближайшая колонка против нормированного пространственного распределения с шириной один шаг решётки. Для B/C/D парные продолжения из одинаковых завершённых весов, одинаковые 500 шагов и пакеты, ядра заморожены. Все train104/26, один GPU-процесс; сохранение до 19:55, независимый предел 20:00. Критерии, ограничения и источники до прогона — `runs/z13_distribution_pilot_20261015/protocol.json`. Dev, HOTA и независимые видео не используются. Если окно не позволит завершить, сохраняется промежуточное состояние без сокращения шага или выборки.\n\n"
    path.write_text(text.replace(marker,marker+entry,1),encoding='utf-8')
    print('Prepared functional pilot',flush=True)

def worker():
    import torch
    from tracking.gpu_lock import GpuLock
    from tracking.z13_checkpoint import RunControl, save_checkpoint, restore_checkpoint, TrainingPaused
    from scripts.z13_train import Episodes
    from scripts.z13_spatial_train import build, predict
    from tracking.z13_readout import run_sequence, to_eye
    from tracking.z13_distribution_target import spatial_targets, distribution_loss
    torch.set_num_threads(2)
    p=read(OUT/'protocol.json')
    for name,h in p['source_sha256'].items(): assert sha(ROOT/name)==h,name
    control=RunControl(OUT,OUT/'stop.request',SAVE_DEADLINE,resume=True)
    with GpuLock('flyvis_fovea_gpu'):
        try:
            control.check()
            fit=Episodes(p['fit']); evaluation=Episodes(p['evaluation'])
            useful=[i for i,r in enumerate(fit.labels) if r[2]>=0]
            for arm in p['arms']:
                for lossname in p['losses']:
                    folder=OUT/arm/lossname; folder.mkdir(parents=True,exist_ok=True)
                    if (folder/'predictions.json').exists(): continue
                    control.check()
                    torch.manual_seed(0); np.random.seed(0); random.seed(0)
                    est,bank,geom,bb,ro,device=build(arm,torch)
                    ck=torch.load(SRC/arm/'full/model.pt',map_location='cpu',weights_only=False)
                    (bb if arm=='B' else bb.net).load_state_dict(ck['backbone']); ro.load_state_dict(ck['readout'])
                    for param in (bb if arm=='B' else bb.net).parameters(): param.requires_grad_(False)
                    opt=torch.optim.Adam(ro.parameters(),lr=p['lr']); rng=np.random.default_rng(p['seed'])
                    ident=dict(protocol_sha256=sha(OUT/'protocol.json'),arm=arm,loss=lossname)
                    checkpoint=folder/'checkpoint.pt'; history=[]; start=0
                    if checkpoint.exists(): start,history=restore_checkpoint(torch,checkpoint,arm,bb,ro,opt,rng,ident)
                    else: save_checkpoint(torch,checkpoint,arm,bb,ro,opt,rng,0,history,ident)
                    points=torch.as_tensor(geom.points,device=device)
                    t=time.monotonic(); ro.train()
                    for step in range(start,p['steps']):
                        control.check(); picked=rng.choice(useful,p['batch'],replace=False); opt.zero_grad(set_to_none=True); value=0.
                        for left in range(0,p['batch'],p['micro']):
                            labels=[fit.labels[i] for i in picked[left:left+p['micro']]]
                            hx,mask=fit.batch(torch,device,[r[0] for r in labels],[r[1] for r in labels])
                            logits=run_sequence(torch,bb,ro,hx,mask)
                            selected=logits[torch.arange(len(labels),device=device),torch.as_tensor([r[1] for r in labels],device=device)]
                            if lossname=='hard':
                                target=torch.as_tensor([r[3] if r[2]==1 else 721 for r in labels],device=device)
                                loss=torch.nn.functional.cross_entropy(selected,target)
                            else:
                                centers=[to_eye(r[4],r[7],bank.eye_size) if r[2]==1 else (0.,0.) for r in labels]
                                target,known=spatial_targets(torch,[r[2] for r in labels],centers,points,geom.spacing)
                                loss=distribution_loss(torch,selected,target,known)
                            if not bool(torch.isfinite(loss)): raise ValueError('Nonfinite loss')
                            (loss*len(labels)/p['batch']).backward(); value+=float(loss.detach())*len(labels)/p['batch']
                        if any(q.grad is not None and not bool(torch.isfinite(q.grad).all()) for q in ro.parameters()): raise ValueError('Nonfinite gradient')
                        opt.step(); history.append(dict(step=step+1,loss=value))
                        if (step+1)%10==0 or step+1==p['steps'] or control.stopped():
                            save_checkpoint(torch,checkpoint,arm,bb,ro,opt,rng,step+1,history,ident)
                            control.status(state='running',stage='training',arm=arm,loss=lossname,step=step+1,total=p['steps'],checkpoint_verified=True,
                                seconds_per_step=(time.monotonic()-t)/(step+1-start))
                        if (step+1)%100==0: print(arm,lossname,step+1,value,flush=True)
                        control.check()
                    current=(bb if arm=='B' else bb.net).state_dict()
                    assert all(torch.equal(v.cpu(),ck['backbone'][k].cpu()) for k,v in current.items()),'Frozen backbone changed'
                    control.status(stage='prediction',arm=arm,loss=lossname)
                    rows=predict(torch,evaluation,bb,ro,device,geom,bank.eye_size,control)
                    write(folder/'predictions.json',rows)
                    write(folder/'completed.json',dict(finished=now().isoformat(),rows=len(rows),steps=p['steps'],backbone_unchanged=True))
                    del bb,ro,est,bank,opt,ck,current; torch.cuda.empty_cache()
            control.status(state='completed',stage='all_predictions_saved')
        except TrainingPaused:
            control.status(state='paused',stage='deadline_or_stop',saved=True)
            return 75
    return 0

def controller():
    deadline=datetime.fromisoformat(DEADLINE); remaining=(deadline-now()).total_seconds()
    assert remaining>0
    with (OUT/'stdout.log').open('a',encoding='utf-8') as out, (OUT/'stderr.log').open('a',encoding='utf-8') as err:
        proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--worker'],cwd=ROOT,
            stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.IDLE_PRIORITY_CLASS)
        write(OUT/'controller.json',dict(pid=os.getpid(),worker_pid=proc.pid,started=now().isoformat(),deadline=DEADLINE,save_deadline=SAVE_DEADLINE))
        try: code=proc.wait(timeout=max(1,(deadline-now()).total_seconds()))
        except subprocess.TimeoutExpired:
            (OUT/'stop.request').write_text('Absolute author deadline',encoding='utf-8')
            subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],creationflags=subprocess.CREATE_NO_WINDOW,check=False)
            code=proc.wait(timeout=30)
        write(OUT/'controller_result.json',dict(exit_code=code,ended=now().isoformat(),own_worker_ended=proc.poll() is not None))

if __name__=='__main__':
    if '--prepare' in sys.argv: prepare()
    elif '--worker' in sys.argv: raise SystemExit(worker())
    else: controller()
