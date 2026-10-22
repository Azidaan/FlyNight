"""One fixed observed target/background loss replacement; no core training."""
from pathlib import Path
from datetime import datetime
import sys
import time
import os
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_center_hota_20261021 import read,write,sha,now
RUN=ROOT/'runs/z13_balanced_ridge_20261022';IDENT='z13_balanced_ridge_v1'


def prepare():
    if RUN.exists():raise RuntimeError('Existing run must be preserved')
    previous=read(ROOT/'runs/z13_absolute_confidence_20261022/report.json');assert not previous['primary_pass']
    data=read(ROOT/'runs/z13_query_memory_20261021/protocol.json')['evaluate']
    files=['tracking/z13_balanced_ridge.py','tests/test_z13_balanced_ridge.py','scripts/z13_balanced_ridge_20261022.py',
        'scripts/z13_bounded_stage_20261022.py','scripts/z13_stage_progress_20261022.ps1',
        'scripts/z13_target_model.py','scripts/z13_train.py','tracking/z13_readout.py','tracking/z13_target_center.py',
        'tracking/z13_ridge_localizer.py','scripts/z13_continue.py','scripts/z13_score_swap.py',
        'tracking/flyvis_motion.py','tracking/foveated_eyes.py','tracking/retina.py','scripts/run_bdd_video_rate_experiment.py',
        'runs/flyvis/results/flow/0000/000/_meta.yaml','runs/flyvis/results/flow/0000/000/best_chkpt',
        'runs/z13_spatial_v2_20261013/D/full/model.pt','runs/z13_query_memory_20261021/protocol.json',
        'runs/z13_absolute_confidence_20261022/report.json']
    for arm in ('D','C','R1'):
        files.extend([f'runs/z13_target_center_dev_20261018/{arm}/normalization.json',f'runs/z13_target_center_20261018/{arm}/choice.json',f'runs/z13_query_memory_20261021/{arm}/last_rows.json'])
    files.extend(f'runs/z13_data/{v}.npz' for v in data)
    p=dict(id=IDENT,created=now().isoformat(),device='cpu',budget_seconds=10800,save_at='2026-10-23T07:35:00+03:00',hard_stop='2026-10-23T07:39:30+03:00',
        authorization='Author22.10: Продолжи; Работай в автономном режиме. Necessary justified long experiments and publication authorized. CPU chosen to avoid competing with active foreground GPU applications.',
        action='Replace uniform spatial least-squares loss by equal total observed-target/background weights in per-target Gaussian ridge fitting. No feature/core/history/target/lambda/score change.',
        weights='Observed mask mass0.5,complement mass0.5; each region uniform. Sum weights1 so ridge regularization keeps its scale. Empty observed masks fail; an all-target mask uses uniform weights. Weighted means/bias are part of the same weighted least-squares objective. No search annotation enters fit.',
        evaluate=data,arms=['D','C','R1'],minimum_useful_shift=.02,
        primary=['Uniform control reproduces every saved8326row per arm: center<0.5px,score<0.001,identical saved lambda. All input hashes fixed.',
            'D balanced-D uniform: mean paired-video useful-center fraction>=0.02 and95%lower>0 on all present train26 rows.',
            'D balanced-D uniform:95%lower>0 for selected_success_fraction_positive at equal1%quota84.',
            'D balanced-R1 balanced:95%lower>0 on all present train26 rows; R1 receives identical history and weighting.'],
        technical_gate='First8Depisodes are saved as first full chunk, not repeated. Check original parity; estimate full padded-frame cost conservatively3*Dcost*1.5 with initialization elapsed added. Proceed only if estimate fits remaining10800s minus60s and author save window. Probe outcome is timing/parity only; do not make an efficacy decision on8episodes.',
        counterargument='Foreground/background already enter the original ridge fit. Balancing may amplify clutter inside the observed box, reduce background discrimination, or improve R1 equally. No claim the original failure was caused by imbalance; no claim D anatomy superiority.',
        decision='Only if all primary gates pass, propose a separate fixed dev transfer/stand trial. Otherwise retain uniform centered readout; stop this weighting implementation and record exact failed gates. No HOTA/core training or dev tuning here. Old90%/50refusal criterion unchanged.',
        alternatives_ranked=['Fixed target/background balance: cheapest closed-form objective change with no fitted hyperparameters.','DiMP-like clipped background residual: separate future objective, requires iterative solver and convergence validation.','PrDiMP conditional density: larger separate change; density over locations alone does not establish target presence.'],
        references=[dict(url='https://arxiv.org/html/1904.07220v2',section='3.1 equations1–2 and3.4',supports='Spatial weights address target/background imbalance; DiMP also learns masks and clips background residuals. Our fixed balanced ridge is only one isolated component, not DiMP reproduction or learned loss.'),dict(url='https://arxiv.org/html/2003.12565v1',section='2.2 and3',supports='Confidence depends on objective; probabilistic alternatives exist. No support for a guaranteed effect or calibrated refusal from our balancing.')],
        sha256={name:sha(ROOT/name) for name in files})
    RUN.mkdir();write(RUN/'protocol.json',p);write(RUN/'status.json',dict(state='ready',stage='Баланс цели и фона: техническая проба, затем полный train26',done=0,total=3,updated=now().isoformat()))
    cat=read(ROOT/'experiments/project_history.json');assert not any(c['id']==IDENT for c in cat['experiments'])
    cat['experiments'].append(dict(id=IDENT,title='З13: равный суммарный вес цели и фона при подгонке модели',area='flyvis',state='ready',started='2026-10-22',done='',tags=['fly-problem-P8'],
        question='Улучшит ли фиксированная балансировка функции ошибки локализацию D и выбор её полезных ответов?',hypothesis='Равный суммарный вес наблюдаемой рамки и фона может изменить различающую модель; причинная роль дисбаланса пока не установлена.',expected='Одна функция ошибки, прежняя гауссова цель,68признаков и lambda; D/C/R1,полные26train,проверенное воспроизведение и парные интервалы. Сначала стоимость/воспроизведение8эпизодов, без вывода об эффективности по пробе.',
        acceptance=[dict(criterion=x,met=None) for x in p['primary']],summary='Критерии и фиксированные веса до расчёта.',obtained='',interpretation='',decision='',cannot_claim='Не воспроизведение DiMP, не причинное доказательство дисбаланса, не калиброванный отказ и не HOTA. Поисковый повторно используемый train26; прежние90%/50не заменяются. Ни ядро, ни detector/ByteTrack/MASA не обучаются и не меняются.',metrics=[],commits=[],files=files[:3],evidence=[dict(kind='file',ref='runs/z13_balanced_ridge_20261022/protocol.json',claim='Одна замена функции ошибки с критериями до исхода')]))
    write(ROOT/'experiments/project_history.json',cat)
    path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';content=path.read_text(encoding='utf-8');anchor='## Журнал изменений\n';assert content.count(anchor)==1
    event=f'\n**{now():%d.%m.%Y, %H:%M} MSK** — постановка взвешенной модели цели `{IDENT}`\n\n*Постановка.* '+p['action']+'\n\n'+p['weights']+'\n\nКритерии до прогона: '+str(p['primary'])+'\n\nКонтраргумент: '+p['counterargument']+'\n\nОпора: DiMP1904.07220v2§3.1 использует пространственные веса для дисбаланса цели/фона. У нас только фиксированный взвешенный узел, не полная DiMP; фон уже учитывался и прежде. Стоимость: '+p['technical_gate']+'\n\nCPUдо10800с, отдельная консоль и независимый контроллер; сохранение не позже23.10 07:35МСК, остановка своего дерева07:39:30. GPU,dev,HOTA,отложенные75не используются. Это новая рабочая версия подгонки считывания после двух не принятых простых замен балла; бесконечного перебора селекторов не будет. Предложения прежних ветвей и старые итоги сохранены.\n'
    path.write_text(content.replace(anchor,anchor+event,1),encoding='utf-8')


def worker():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    torch.set_num_threads(2);torch.set_default_device('cpu')
    assert not torch.cuda.is_available()
    from scripts import z13_target_model as old
    from scripts.z13_train import Episodes,success
    from tracking.z13_target_center import observed_center_target
    from tracking.z13_balanced_ridge import fit_observed
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
    verify();status('Загрузка фиксированных эпизодов',done=0,total=3)
    data=Episodes(p['evaluate']);by_ep={};videos=[]
    for v,n in data.meta:videos.extend([v]*n)
    for i,l in enumerate(data.labels):
        if not l[8]:by_ep.setdefault(l[0],[]).append(i)
    eps=sorted(by_ep);groups=[eps[left:left+8] for left in range(0,len(eps),8)]
    maxima={e:max(data.labels[i][1] for i in by_ep[e]) for e in eps}
    cost=[len(group)*(5+max(maxima[e] for e in group)) for group in groups]
    checkpoint=torch.load(ROOT/'runs/z13_spatial_v2_20261013/D/full/model.pt',map_location='cpu',weights_only=False);allrows={};repro={}
    for ai,arm in enumerate(p['arms']):
        folder=RUN/arm
        if (folder/'done.json').exists():
            for mode in ('uniform','balanced'):allrows[arm+'_'+mode]=read(folder/(mode+'_rows.json'))
            repro[arm]=read(folder/'reproduction.json');continue
        status('Сборка фиксированной модели, CPU',arm=arm,done=ai,total=3)
        bank,geom,bb=old.build_arm(arm,torch,checkpoint)
        assert bb is None or str(bb.dev)=='cpu'
        normal=read(ROOT/f'runs/z13_target_center_dev_20261018/{arm}/normalization.json');stats=(np.asarray(normal['mean']),np.asarray(normal['sd']))
        lam=read(ROOT/f'runs/z13_target_center_20261018/{arm}/choice.json')['center']['lambda_value']
        saved={r['label']:r for r in read(ROOT/f'runs/z13_query_memory_20261021/{arm}/last_rows.json')};assert len(saved)==8326
        rows={m:[] for m in ('uniform','balanced')};dc=ds=0.;fallbacks=0
        for gi,group in enumerate(groups):
            chunk=folder/f'chunks/{gi:05d}.json'
            if chunk.exists():
                result=read(chunk)
                assert sha(chunk)==read(chunk.with_suffix('.sha256.json'))['sha256'] and result['episodes']==group
            else:
                status('Подгонка цели/фона и неизменный контроль',arm=arm,arms_done=ai,done=gi,total=len(groups),eta='unknown')
                began=time.monotonic();hx,masks=data.batch(torch,'cpu',group,[maxima[e] for e in group]);needed={3}|{4+data.labels[i][1] for e in group for i in by_ep[e]}
                features=old.episode_features(torch,arm,bb,hx,geom.neighbors,needed)
                result=dict(episodes=group,rows={m:[] for m in rows},max_center_px=0.,max_score=0.,full_mask_fallbacks=0)
                for j,e in enumerate(group):
                    check();mask=masks[j].cpu().numpy().astype(bool);result['full_mask_fallbacks']+=int(mask.all());target=observed_center_target(mask,geom.points,geom.spacing)
                    xe=(features[3][j].T-stats[0])/stats[1]
                    models={m:fit_observed(xe,target,mask,lam,m=='balanced') for m in rows}
                    for i in by_ep[e]:
                        l=data.labels[i];xs=(features[4+l[1]][j].T-stats[0])/stats[1]
                        for mode,model in models.items():
                            center,z=map_center(ridge_response(model,xs),geom);center=np.asarray(to_frame(center,l[7],bank.eye_size));r=dict(label=i,video=videos[e],kind=int(l[2]),pos=int(l[1]),p_present=z,center=center.tolist(),pred=np.asarray(l[6]).tolist())
                            if l[2]==1:
                                r.update(err=float(np.linalg.norm(center-l[4])),err_pred=float(np.linalg.norm(np.asarray(l[6])-l[4])),scale=float(max(l[5][2]-l[5][0],l[5][3]-l[5][1])));r['useful']=bool(success(r))
                            elif l[2]==0:r['useful']=False
                            result['rows'][mode].append(r)
                            if mode=='uniform':
                                ref=saved[i];assert ref['video']==r['video'] and ref['kind']==r['kind']
                                result['max_center_px']=max(result['max_center_px'],float(np.max(np.abs(center-ref['center']))));result['max_score']=max(result['max_score'],abs(z-ref['p_present']))
                result['seconds']=time.monotonic()-began;assert result['max_center_px']<.5 and result['max_score']<.001
                write(chunk,result);write(chunk.with_suffix('.sha256.json'),dict(sha256=sha(chunk)))
                del features,hx,masks
            for mode in rows:rows[mode].extend(result['rows'][mode])
            dc=max(dc,result['max_center_px']);ds=max(ds,result['max_score']);fallbacks+=result['full_mask_fallbacks']
            if arm=='D' and gi==0 and not (RUN/'technical.json').exists():
                estimate=3*1.5*sum(cost)/cost[0]*result['seconds']
                remaining=min(p['budget_seconds']-(time.monotonic()-started),(datetime.fromisoformat(p['save_at'])-now()).total_seconds())
                technical=dict(at=now().isoformat(),device='cpu',episodes=group,probe_seconds=result['seconds'],max_center_px=result['max_center_px'],max_score=result['max_score'],padded_frames=sum(cost),probe_padded_frames=cost[0],estimated_remaining_seconds=estimate,available_seconds=remaining,proceed=estimate+60<remaining,meaning='Timing/parity only. No efficacy verdict on these8episodes. First chunk retained.')
                write(RUN/'technical.json',technical)
                if not technical['proceed']:
                    write(RUN/'status.json',dict(state='completed',stage='Проба сохранена, полный расчёт не помещается в бюджет',done=0,total=3,updated=now().isoformat()));return
        assert len(rows['uniform'])==len(rows['balanced'])==8326
        for mode in rows:write(folder/(mode+'_rows.json'),rows[mode]);allrows[arm+'_'+mode]=rows[mode]
        repro[arm]=dict(rows=8326,max_center_px=dc,max_score=ds,lambda_value=lam,full_mask_fallbacks=fallbacks);write(folder/'reproduction.json',repro[arm]);write(folder/'done.json',dict(at=now().isoformat(),uniform_rows_sha256=sha(folder/'uniform_rows.json'),balanced_rows_sha256=sha(folder/'balanced_rows.json')))
        del bb,bank
    pv={name:per_video_useful(rows) for name,rows in allrows.items()};selected={};summary={}
    for name,rows in allrows.items():
        chosen=selected_keys(rows,.01);assert len(chosen)==84
        selected[name]=per_video_useful([dict(r,useful=bool(r.get('useful')) and key(r) in chosen) for r in rows]);s=[r for r in rows if key(r) in chosen];known=[r for r in s if r['kind']>=0]
        summary[name]=dict(rows=len(rows),useful_mean_video=float(np.mean(list(pv[name].values()))),coverage=coverage_table(rows),selected=84,known=len(known),unknown=84-len(known),selected_useful=sum(bool(r.get('useful')) for r in known),selected_absent=sum(r['kind']==0 for r in s))
    comparisons={a+'-'+b:paired(pv[a],pv[b]) for a,b in [(arm+'_balanced',arm+'_uniform') for arm in p['arms']]+[('D_balanced','R1_balanced'),('C_balanced','R1_balanced'),('D_balanced','C_balanced')]}
    selected_pairs={arm+'_balanced-'+arm+'_uniform':paired(selected[arm+'_balanced'],selected[arm+'_uniform']) for arm in p['arms']}
    d=comparisons['D_balanced-D_uniform'];criteria=dict(reproduction=True,localization=d['mean_per_video']>=p['minimum_useful_shift'] and d['ci95'][0]>0,selection=selected_pairs['D_balanced-D_uniform']['ci95'][0]>0,fly_control=comparisons['D_balanced-R1_balanced']['ci95'][0]>0)
    verify();write(RUN/'reproduction.json',repro);write(RUN/'report.json',dict(primary_pass=all(criteria.values()),criteria=criteria,paired=comparisons,selected_paired=selected_pairs,per_video=pv,selected_per_video=selected,summary=summary,device='cpu',hota_measured=False,dev_evaluated=False,limitations='One fixed observed-only weight change, frozen lambda/features/core; repeated exploratory train26. Not DiMP reproduction, not calibrated refusal; no HOTA. Old90%/50criterion unchanged.'))
    write(RUN/'status.json',dict(state='completed',stage='Полный train26, все контроли и исходы сохранены',done=3,total=3,updated=now().isoformat()))


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--worker' in sys.argv:
        try:worker()
        except Exception as e:write(RUN/'status.json',dict(state='failed',stage=str(e),updated=now().isoformat()));raise
    else:raise RuntimeError('Explicit prepare/worker required')
