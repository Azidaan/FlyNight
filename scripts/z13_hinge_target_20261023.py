"""One isolated observed-only residual change with fixed balanced ridge control."""
from pathlib import Path
from datetime import datetime
import sys
import time
import os
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_center_hota_20261021 import read,write,sha,now
RUN=ROOT/'runs/z13_hinge_target_20261023';IDENT='z13_hinge_target_v1';MODES=('balanced','hinge')


def prepare():
    if RUN.exists():raise RuntimeError('Existing run must be preserved')
    previous=ROOT/'runs/z13_balanced_ridge_20261022';old=read(previous/'protocol.json');assert read(previous/'full_finish_recorded.json')['complete']
    files=['tracking/z13_hinge_target.py','tests/test_z13_hinge_target.py','scripts/z13_hinge_target_20261023.py','scripts/z13_bounded_recovery_controller_20261022.py',
        'scripts/z13_hinge_progress_20261023.ps1','runs/z13_balanced_ridge_20261022/protocol.json','runs/z13_balanced_ridge_20261022/report.json']
    files.extend(f'runs/z13_balanced_ridge_20261022/{arm}/balanced_rows.json' for arm in old['arms'])
    p=dict(old);p.update(id=IDENT,created=now().isoformat(),budget_seconds=7200,stop_request_path='runs/z13_hinge_target_20261023/stop.request',
        action='Change only background residual in observed per-target fitting: from s-y to max(0,s-y) outside observed mask. Foreground s-y,Gaussian targets,balanced0.5/0.5weights,features,core,lambda,history and peak-z readout unchanged.',
        method='Convex squared-hinge variant with fixed Gaussian upper bound on background response. It differs explicitly from DiMP max(0,s)-y. Initialize at identical balanced ridge; damped active-set Newton,Armijo1e-4,100iterations,31backtracking steps,gradient infinity norm<=1e-8*(1+initial norm). No criterion/solver tuning on evaluation labels.',
        primary=['Reproduce previous balanced control for all8326rows/arm: center<0.5px,score<0.001,same lambda. Every observed solver converges at frozen tolerance; hashes unchanged.',
            'D hinge-D balanced mean paired-video useful-center fraction>=0.02 and95%lower>0 on all present train26 rows.',
            'D hinge-D balanced95%lower>0 for selected_success_fraction_positive at equal1%quota84.',
            'D hinge-R1 hinge95%lower>0 on all present train26 rows, identical history and objective.'],
        technical_gate='Save first8Depisodes as first full chunk,check baseline parity and every observed solver convergence. Estimate full padded-frame work3*1.5*Dprobe_time; proceed only inside original7200s minus60s and Moscow save window. No efficacy decision on probe.',
        counterargument='Balanced D improved localization but not refusal:2useful of84. A one-sided background residual may free model capacity, but also increase false peaks or leave target absence unresolved. The feature core is not proven responsible for rejection failure. Same no-FlyVis R1 and balanced control are compulsory.',
        decision='Only if all main gates pass proceed to a separate fixed transfer/stand protocol. Otherwise stop this one-sided residual implementation; retain accepted prior version and record localization versus refusal separately. No endless scalar selector search and no core training,dev,HOTA,protected75 here. Old90%/50criterion unchanged.',
        alternatives_ranked=['Current:one-sided background residual,changes target-model objective and has a matched balanced control.','Conditional density with explicit absence training:separate larger design; PrDiMP density over locations alone does not solve presence.','More core training without an improved objective:deferred; prior continuation did not improve and present refusal criterion failed.'],
        references=[dict(url='https://arxiv.org/html/1904.07220v2',section='3.1,equations1–2',supports='DiMP permits low/negative background scores through a clipped residual. Our max(0,s-y) keeps the exact prior Gaussian target as an upper bound and remains convex; it is not a full DiMP reproduction or performance transfer.')],
        sha256={**old['sha256'],**{name:sha(ROOT/name) for name in files}})
    RUN.mkdir();write(RUN/'protocol.json',p);write(RUN/'status.json',dict(state='ready',stage='Односторонняя ошибка фона: проба, затем полный train26',done=0,total=3,updated=now().isoformat()))
    cat=read(ROOT/'experiments/project_history.json');assert not any(c['id']==IDENT for c in cat['experiments'])
    cat['experiments'].append(dict(id=IDENT,title='З13: односторонняя ошибка фона в модели цели',area='flyvis',state='ready',started=f'{now():%Y-%m-%d}',done='',tags=['fly-problem-P8'],
        question='Поможет ли прекращение штрафа уже подавленного фона улучшить D сверх согласованной балансировки, включая отбор полезных ответов?',
        hypothesis='Освобождение от регрессии точных низких значений фона может улучшить различающую модель; исправление отказа не установлено.',expected='D/C/R1,полные26train,одна функция ошибки; прежние гауссова цель,веса0,5/0,5,68признаков,история,коэффициент регуляризации. CPUдо7200с,сначала стоимость/воспроизведение.',
        acceptance=[dict(criterion=c,met=None) for c in p['primary']],summary='До расчёта зарегистрирована отдельная замена остаточной ошибки фона.',obtained='',interpretation='',decision='',cannot_claim='Не полная DiMP и не её точная формула; не HOTA,не калиброванный отказ,не обучение734параметров. train26 используется повторно,старые90%/50не заменены.',metrics=[],commits=[],files=files[:3],evidence=[dict(kind='file',ref='runs/z13_hinge_target_20261023/protocol.json',claim='Одна замена с критериями до исхода'),dict(kind='file',ref='runs/z13_balanced_ridge_20261022/report.json',claim='Обязательный согласованный контроль и непройденный отказ')]))
    write(ROOT/'experiments/project_history.json',cat)
    journal('постановка односторонней ошибки фона','*Постановка.* '+p['action']+'\n\n'+p['method']+'\n\nКритерии до исхода: '+str(p['primary'])+'\n\nКонтраргумент: '+p['counterargument']+'\n\nОпора: DiMP1904.07220v2§3.1; наша выпуклая версия max(0,s-y) сохраняет прежнюю гауссову цель и отличается от max(0,s)-y в статье. Это одна функция ошибки узла считывания, не внешний фильтр. CPUдо7200с,техническая проба до полного26train; сохранение23.10 07:35МСК, полное окончание своего дерева07:39:30. У контроллера действует ещё более ранний двухчасовой предел. Никаких новых загрузок,dev,HOTA,отложенных75или обучения ядра. Предыдущий полный исход сохранён и опубликован; следующий критерий не ослабляется из-за частного положительного эффекта.')


def journal(title,body):
    path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';text=path.read_text(encoding='utf-8');anchor='## Журнал изменений\n';assert text.count(anchor)==1
    path.write_text(text.replace(anchor,anchor+f'\n**{now():%d.%m.%Y, %H:%M} MSK** — {title} `{IDENT}`\n\n'+body+'\n',1),encoding='utf-8')


def worker():
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import torch
    torch.cuda.is_available=lambda:False;torch.cuda.device_count=lambda:0
    torch.set_num_threads(2);torch.set_default_device('cpu');assert str(torch.ones(1,device='cpu').device)=='cpu'
    from scripts import z13_target_model as old
    from scripts.z13_train import Episodes,success
    from tracking.z13_target_center import observed_center_target
    from tracking.z13_balanced_ridge import fit_observed
    from tracking.z13_hinge_target import fit_hinge
    from tracking.z13_ridge_localizer import ridge_response,map_center
    from tracking.z13_readout import to_frame
    from scripts.z13_continue import paired,per_video_useful,coverage_table
    from scripts.z13_score_swap import selected_keys,key
    p=read(RUN/'protocol.json');started=time.monotonic()
    def check():
        if time.monotonic()-started>p['budget_seconds']-30 or (RUN/'stop.request').exists() or now()>=datetime.fromisoformat(p['save_at']):raise TimeoutError('Bounded CPU/author window')
    def verify():
        for name,digest in p['sha256'].items():check();assert sha(ROOT/name)==digest,name
    def status(stage,**kw):
        check();write(RUN/'status.json',dict(state='running',stage=stage,elapsed_seconds=time.monotonic()-started,updated=now().isoformat(),**kw))
    verify();status('Загрузка фиксированных эпизодов',done=0,total=3);data=Episodes(p['evaluate']);by_ep={};videos=[]
    for v,n in data.meta:videos.extend([v]*n)
    for i,label in enumerate(data.labels):
        if not label[8]:by_ep.setdefault(label[0],[]).append(i)
    eps=sorted(by_ep);groups=[eps[left:left+8] for left in range(0,len(eps),8)];maxima={e:max(data.labels[i][1] for i in by_ep[e]) for e in eps};cost=[len(g)*(5+max(maxima[e] for e in g)) for g in groups]
    checkpoint=torch.load(ROOT/'runs/z13_spatial_v2_20261013/D/full/model.pt',map_location='cpu',weights_only=False);allrows={};reproduction={}
    for ai,arm in enumerate(p['arms']):
        folder=RUN/arm
        if (folder/'done.json').exists():
            for mode in MODES:allrows[arm+'_'+mode]=read(folder/(mode+'_rows.json'))
            reproduction[arm]=read(folder/'reproduction.json');continue
        status('Сборка фиксированной модели CPU',arm=arm,arms_done=ai,done=0,total=len(groups));bank,geom,bb=old.build_arm(arm,torch,checkpoint);assert bb is None or str(bb.dev)=='cpu'
        normal=read(ROOT/f'runs/z13_target_center_dev_20261018/{arm}/normalization.json');stats=(np.asarray(normal['mean']),np.asarray(normal['sd']));lam=read(ROOT/f'runs/z13_target_center_20261018/{arm}/choice.json')['center']['lambda_value']
        saved={r['label']:r for r in read(ROOT/f'runs/z13_balanced_ridge_20261022/{arm}/balanced_rows.json')};assert len(saved)==8326
        rows={mode:[] for mode in MODES};dc=ds=0.;iterations=gradient_ratio=0.;solvers=0
        for gi,group in enumerate(groups):
            chunk=folder/f'chunks/{gi:05d}.json'
            if chunk.exists():
                result=read(chunk);assert sha(chunk)==read(chunk.with_suffix('.sha256.json'))['sha256'] and result['episodes']==group
            else:
                status('Односторонняя ошибка фона и согласованный контроль',arm=arm,arms_done=ai,done=gi,total=len(groups),eta='unknown');began=time.monotonic()
                hx,masks=data.batch(torch,'cpu',group,[maxima[e] for e in group]);needed={3}|{4+data.labels[i][1] for e in group for i in by_ep[e]};features=old.episode_features(torch,arm,bb,hx,geom.neighbors,needed)
                result=dict(episodes=group,rows={mode:[] for mode in MODES},max_center_px=0.,max_score=0.,solver_max_iterations=0,solver_max_gradient_ratio=0.,solvers=0)
                for j,e in enumerate(group):
                    check();mask=masks[j].cpu().numpy().astype(bool);target=observed_center_target(mask,geom.points,geom.spacing);xe=(features[3][j].T-stats[0])/stats[1]
                    candidate,info=fit_hinge(xe,target,mask,lam);assert info['monotone'] and info['gradient_max']<=info['tolerance']
                    models=dict(balanced=fit_observed(xe,target,mask,lam,True),hinge=candidate);result['solvers']+=1;result['solver_max_iterations']=max(result['solver_max_iterations'],info['iterations']);result['solver_max_gradient_ratio']=max(result['solver_max_gradient_ratio'],info['gradient_max']/info['tolerance'])
                    for i in by_ep[e]:
                        label=data.labels[i];xs=(features[4+label[1]][j].T-stats[0])/stats[1]
                        for mode,model in models.items():
                            center,z=map_center(ridge_response(model,xs),geom);center=np.asarray(to_frame(center,label[7],bank.eye_size));r=dict(label=i,video=videos[e],kind=int(label[2]),pos=int(label[1]),p_present=z,center=center.tolist(),pred=np.asarray(label[6]).tolist())
                            if label[2]==1:
                                r.update(err=float(np.linalg.norm(center-label[4])),err_pred=float(np.linalg.norm(np.asarray(label[6])-label[4])),scale=float(max(label[5][2]-label[5][0],label[5][3]-label[5][1])));r['useful']=bool(success(r))
                            elif label[2]==0:r['useful']=False
                            result['rows'][mode].append(r)
                            if mode=='balanced':
                                ref=saved[i];assert ref['video']==r['video'] and ref['kind']==r['kind'];result['max_center_px']=max(result['max_center_px'],float(np.max(np.abs(center-ref['center']))));result['max_score']=max(result['max_score'],abs(z-ref['p_present']))
                result['seconds']=time.monotonic()-began;assert result['max_center_px']<.5 and result['max_score']<.001
                write(chunk,result);write(chunk.with_suffix('.sha256.json'),dict(sha256=sha(chunk)));del features,hx,masks
            for mode in MODES:rows[mode].extend(result['rows'][mode])
            dc=max(dc,result['max_center_px']);ds=max(ds,result['max_score']);iterations=max(iterations,result['solver_max_iterations']);gradient_ratio=max(gradient_ratio,result['solver_max_gradient_ratio']);solvers+=result['solvers']
            if ai==0 and gi==0 and not (RUN/'technical.json').exists():
                estimate=3*1.5*sum(cost)/cost[0]*result['seconds'];remaining=min(p['budget_seconds']-(time.monotonic()-started),(datetime.fromisoformat(p['save_at'])-now()).total_seconds());technical=dict(at=now().isoformat(),device='cpu',episodes=group,probe_seconds=result['seconds'],max_center_px=dc,max_score=ds,solver_max_iterations=iterations,solver_max_gradient_ratio=gradient_ratio,estimated_remaining_seconds=estimate,available_seconds=remaining,proceed=estimate+60<remaining,meaning='Timing,solver and parity only; no efficacy verdict on8episodes. First saved chunk retained.')
                write(RUN/'technical.json',technical)
                if not technical['proceed']:write(RUN/'status.json',dict(state='completed',stage='Проба сохранена, полный расчёт не помещается в бюджет',done=0,total=3,updated=now().isoformat()));return
        assert len(rows['balanced'])==len(rows['hinge'])==8326
        for mode in MODES:write(folder/(mode+'_rows.json'),rows[mode]);allrows[arm+'_'+mode]=rows[mode]
        reproduction[arm]=dict(rows=8326,max_center_px=dc,max_score=ds,lambda_value=lam,solver_max_iterations=iterations,solver_max_gradient_ratio=gradient_ratio,solvers=solvers);write(folder/'reproduction.json',reproduction[arm]);write(folder/'done.json',dict(at=now().isoformat(),sha256={mode:sha(folder/(mode+'_rows.json')) for mode in MODES}));del bb,bank
    pv={name:per_video_useful(rows) for name,rows in allrows.items()};selected={};summary={}
    for name,rows in allrows.items():
        chosen=selected_keys(rows,.01);assert len(chosen)==84;selected[name]=per_video_useful([dict(r,useful=bool(r.get('useful')) and key(r) in chosen) for r in rows]);s=[r for r in rows if key(r) in chosen];known=[r for r in s if r['kind']>=0]
        summary[name]=dict(rows=len(rows),useful_mean_video=float(np.mean(list(pv[name].values()))),coverage=coverage_table(rows),selected=84,known=len(known),unknown=84-len(known),selected_useful=sum(bool(r.get('useful')) for r in known),selected_absent=sum(r['kind']==0 for r in s))
    pairs=[(arm+'_hinge',arm+'_balanced') for arm in p['arms']]+[('D_hinge','R1_hinge'),('C_hinge','R1_hinge'),('D_hinge','C_hinge')];comparisons={a+'-'+b:paired(pv[a],pv[b]) for a,b in pairs};selected_pairs={arm+'_hinge-'+arm+'_balanced':paired(selected[arm+'_hinge'],selected[arm+'_balanced']) for arm in p['arms']}
    d=comparisons['D_hinge-D_balanced'];criteria=dict(reproduction=True,localization=d['mean_per_video']>=p['minimum_useful_shift'] and d['ci95'][0]>0,selection=selected_pairs['D_hinge-D_balanced']['ci95'][0]>0,fly_control=comparisons['D_hinge-R1_hinge']['ci95'][0]>0)
    verify();write(RUN/'reproduction.json',reproduction);write(RUN/'report.json',dict(primary_pass=all(criteria.values()),criteria=criteria,paired=comparisons,selected_paired=selected_pairs,per_video=pv,selected_per_video=selected,summary=summary,device='cpu',hota_measured=False,dev_evaluated=False,limitations='Single background residual change versus matched balanced control; all Gaussian labels,history,features,core and lambda fixed. Not exact DiMP. Repeated exploratory train26; no calibrated absence or HOTA claim; old90%/50criterion unchanged.'));write(RUN/'status.json',dict(state='completed',stage='Все3варианта и полный train26 сохранены',done=3,total=3,updated=now().isoformat()))


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--worker' in sys.argv:
        try:worker()
        except Exception as error:write(RUN/'status.json',dict(state='failed',stage=repr(error),updated=now().isoformat()));raise
    else:raise RuntimeError('Explicit prepare/worker required')
