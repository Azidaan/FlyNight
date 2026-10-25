"""Paired working prototype: target-response proposal -> center vs complete XYWH."""
from pathlib import Path
from datetime import datetime,timedelta
import inspect
import hashlib
import os
import sys
import time
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts.z13_localization_labels_20261023 import verify_sources
from scripts.z13_continue import paired
RUN=ROOT/'runs/z13_box_proposal_20261025'
OLD=ROOT/'runs/z13_assignment_effect_20261024/recovery1'
MAPS=ROOT/'runs/z13_query_memory_20261021'
IDENT='z13_box_proposal_v1'
ARMS=tuple(a+'_'+kind for a in ('D','C','R1','geometry') for kind in ('center','box'))
ALPHAS=(0.,.25,.5,1.)


def journal(title,text):
    from scripts import z13_candidate_reserve_20261021 as m
    old=m.IDENT
    try:m.IDENT=IDENT;m.journal(title,text)
    finally:m.IDENT=old


def source():
    from scripts.z13_region_readout_20261025 import replay_source
    s=replay_source();old='new_cost_rows(q,costs[ri],detections)'
    assert s.count(old)==1
    return s.replace(old,'new_cost_rows(q,costs[ri],detections,t,self.get_dists)')


def contrasts(values):
    videos=list(values['D_box']);out={'Dbox-baseline':paired(values['D_box'],{v:0. for v in videos})}
    for name in ('Dold','D_center','C_box','R1_box','geometry_box'):
        out['Dbox-'+name]=paired(values['D_box'],values[name])
    for name in ('C','R1'):
        out['box_effect_D-'+name]=paired({v:values['D_box'][v]-values['D_center'][v] for v in videos},
            {v:values[name+'_box'][v]-values[name+'_center'][v] for v in videos})
    return out


def primary(c):
    return c['Dbox-baseline']['mean_per_video']>=.02 and all(x['ci95'][0]>0 for x in c.values())


def prepare():
    assert not RUN.exists()
    parent=ROOT/'runs/z13_proposal_geometry_20261025'
    assert read(parent/'recovery1/finish_recorded.json')['complete']
    assert read(parent/'recovery1/independent_audit.json')['partition895_and_paired_recomputed']
    gate=read(parent/'recovery1/report.json');assert gate['admit_full_box_prototype']
    old=read(OLD/'protocol.json');fit=old['fit20'];ev=old['evaluate26']
    names=['scripts/z13_box_proposal_20261025.py','tracking/z13_box_proposal.py','tests/test_z13_box_proposal.py',
        'tracking/z13_region_readout.py','tracking/z13_proposal_geometry.py','tracking/z13_readout.py',
        'tracking/z13_ridge_localizer.py','scripts/z13_region_readout_20261025.py','scripts/z13_query_memory_20261021.py',
        'scripts/z13_assignment_effect_20261024.py','scripts/z13_bounded_recovery_controller_20261022.py',
        'runs/z13_local_peak_20261021/geometry.json','runs/z13_center_hota_20261021/D/config.json',
        (parent/'recovery1/report.json').relative_to(ROOT).as_posix(),(parent/'recovery1/independent_audit.json').relative_to(ROOT).as_posix()]
    for split in ('fit','evaluate'):
        folder=ROOT/'runs/z13_candidate_reserve_20261021'/split
        names.append((folder/'labels.json').relative_to(ROOT).as_posix())
        for a in ('D','C','R1'):
            names += [p.relative_to(ROOT).as_posix() for p in (folder/a).glob('cache_*.npz')]
            names += [p.relative_to(ROOT).as_posix() for p in (MAPS/a/split).glob('maps_*.*')]
            names.append((MAPS/a/split/'labels.json').relative_to(ROOT).as_posix())
    for v in fit+ev:
        names += [f'runs/train_cache/{v}/detections/raw_video.npz',f'runs/train_cache/{v}/alignment.json',
            f'runs/fly_signatures/{v}/signatures.npz',f'data/bdd100k/labels/box_track_20/train/{v}.json',
            (OLD/'videos'/(v+'.json')).relative_to(ROOT).as_posix()]
    start=now();auth=read(ROOT/'runs/z13_weekend_20261024/authorization.json')
    stop=min(start+timedelta(seconds=1200),datetime.fromisoformat(auth['save_at']))
    p=dict(id=IDENT,created=start.isoformat(),timezone='Europe/Moscow',device='cpu',budget_seconds=1200,
        save_at=stop.isoformat(),hard_stop=(stop+timedelta(seconds=20)).isoformat(),stop_request_path=(RUN/'stop.request').relative_to(ROOT).as_posix(),
        fit20=fit,evaluate26=ev,videos=fit+ev,arms=list(ARMS),alphas=list(ALPHAS),
        action='Change the internal learned target-model output from a peak center to an observed detection XYWH proposal, then use the unchanged tracker distance. The target-conditioned ridge map is learned from the same observed exemplar and reused frozen. No offline new core/readout training; a working output-decoder prototype.',
        method='Read unchanged final-query C/D/R1 maps. Project every actual high same-class detection into original eye crop. Retain center-in-crop regions containing at least one observed hex column. Region logit is the existing centered/scaled map mean inside that rectangle, NULLlogit0. Choose argmax including NULL, with NULL winning ties. Chosen real detection yields target XYWH; interpolate original mean toward it using alpha. Center-only twin changes only XY; box twin changes XYWH. Same selection/support/map/calibration/other state for each twin. Standard get_dists and full assignment are unchanged.',
        controls='D/C/R1 x center/box plus identical geometry-only centered negative squared-distance field around original prediction. Dold is original D peak action with the saved presence tau, canonical center and score, never retuned. Alpha0 is exact no-action in every arm. Unsupported/NULL remains original state. All unknown rows retained. No true box, identity or future input enters the decoder.',
        calibration='Fit20 only, independently for each of8arms, alpha from[0,.25,.5,1] maximizes mean-video(beneficial-harmful)/known. Alpha0 first wins ties. Freeze selected indices in choice.json before any evaluation26 replay. No quota/score threshold/size/seed grid. Report all fit and evaluate alpha results.',
        criteria='Dbox-baseline mean>=2pp and every paired95%video lower>0: Dbox versus baseline, saved Dold, same Dcenter, Cbox,R1box,geometrybox; plus differential box effect D-C and D-R1. All required. Partial positive contrast preserved but cannot promote failed full criterion to HOTA.',
        technical='All46raw baseline outputs bit-exact, all8326evaluate and all fit canonical C/D/R1 map top1 centers<0.5px/scores<0.001, unchanged crops/pred/frame/label binding. Dold semantic outcomes exactly equal old replay for its selected top1, or neutral if refused. Complete mean restored, alpha0 no-action exact. All source hashes. First2fit conservative2*mean*44+90 must fit original1200s; completed videos not repeated.',
        difference='Prior region mean/max prototypes replaced costs with categorical probability penalties; they did not move complete XYWH while retaining standard IoU/confidence costs. Prior size regressor predicted sizes directly; this version transfers an actual observed region selected by the same target-response model, with an otherwise-identical center-only twin. New admission is full-box minus center oracle +2.5419pp [1.8492;3.2842] in parent recovery1/report.json, not a claim of learned gain. No failed loss/normalization/state change is combined.',
        counterargument='Selecting real detector boxes can merely duplicate existing detector geometry and drift to neighbors. Region means/normalized confidence already failed in other interfaces, so this deliberately minimal intervention may still fail. Map features may lack identity/box-quality evidence. Equal eye and geometry-only controls must isolate FlyVis; even positive train26 shadow results require separate frozen closed-loop protocol.',
        literature='ATOM https://arxiv.org/html/1811.07628 sections3.1,4.2 separates target localization from complete box estimation. SiamFC++ https://arxiv.org/html/1911.06188 section3; SiamMOT https://arxiv.org/html/2105.11595 sections3,5.1. This reduced observed-box output is not their IoU prediction network or reported improvement.',
        next='Pass -> separately preregister runtime parity/price and matched closed-loop HOTA with no-FlyVis/C/R1 controls. Fail -> keep this exact readout negative; no alpha/NULL/pooling sweeps on26. Record what succeeded diagnostically versus what worked in the actual prototype, and select the next genuinely different node with evidence.',
        limits='CPU1200s+20, hidden independent controller. No GPU/new features/core training/download/dev/HOTA/75. Same reused train26, sparse labels, baseline-state one-step decisions only. Core trained on different objective; no assumption that named paper solves our task. No novelty/absolute stand benefit from oracle or conditional center result.',
        executed_source_sha256=hashlib.sha256(source().encode()).hexdigest(),sha256={n:sha(ROOT/n) for n in sorted(set(names))})
    write(RUN/'protocol.json',p);write(RUN/'status.json',dict(state='ready',stage='Отклик мухи выбирает центр или полную рамку',done=0,total=46,updated=now().isoformat()))
    cat=read(ROOT/'experiments/project_history.json');assert not any(c['id']==IDENT for c in cat['experiments'])
    cat['experiments'].append(dict(id=IDENT,title='Выход модели цели FlyVis: реальная рамка вместо одного центра',area='flyvis',state='ready',started=start.strftime('%Y-%m-%d'),tags=['fly-problem-P8','fly-problem-P9'],question='Даёт ли полная геометрия предложенной области пользу сверх того же центра и равного считывания без мухи?',hypothesis='Подтверждён геометрический запас действия полной рамки. Рабочая гипотеза: обученная на прежнем образце модель цели способна выбрать область достаточно хорошо, чтобы этот запас использовать.',expected=p['action']+' '+p['method'],acceptance=[dict(criterion=p['technical'],met=None),dict(criterion=p['criteria'],met=None)],obtained='',interpretation='',decision='',cannot_claim=p['counterargument']+' '+p['limits'],evidence=[dict(kind='file',ref=(RUN/'protocol.json').relative_to(ROOT).as_posix(),claim='Одна рабочая замена выхода, равный центр/рамка и критерии до исполнения')]))
    write(ROOT/'experiments/project_history.json',cat)
    journal('постановка рабочей версии полного выхода рамки','*Действие.* '+p['action']+'\n\n*Механизм.* '+p['method']+'\n\n*Контроли.* '+p['controls']+'\n\n*Критерии до расчёта.* '+p['criteria']+' '+p['calibration']+'\n\n*Отличие от испытанного.* '+p['difference']+'\n\n*Возражение.* '+p['counterargument']+'\n\n*Источники.* '+p['literature']+'\n\nCPU1200с+20с, скрытый независимый контроллер. '+p['next'])


def aggregate(videos,choices=None):
    values={a:{i:{} for i in range(len(ALPHAS))} for a in ARMS};values['Dold']={0:{}}
    counts={a:dict(known=0,unknown=0,beneficial=0,harmful=0) for a in (*ARMS,'Dold')}
    for v in videos:
        data=read(RUN/'videos'/(v+'.json'));rows=data['rows'];known=[r for r in rows if r['kind']>=0]
        assert known
        for a in values:
            indices=range(len(ALPHAS)) if a!='Dold' else (0,)
            for i in indices:
                actual_index=i if a!='Dold' else 1
                values[a][i][v]=float(np.mean([int(r['arms'][a][actual_index]['beneficial'])-int(r['arms'][a][actual_index]['harmful']) for r in known]))
            chosen=0 if a=='Dold' else (choices[a] if choices is not None else 0)
            ai=1 if a=='Dold' else chosen
            for r in rows:
                if r['kind']<0:counts[a]['unknown']+=1
                else:
                    counts[a]['known']+=1;counts[a]['beneficial']+=bool(r['arms'][a][ai]['beneficial']);counts[a]['harmful']+=bool(r['arms'][a][ai]['harmful'])
    return values,counts


def worker():
    from scripts import z13_assignment_effect_20261024 as original
    from scripts.z13_query_memory_20261021 import load_maps
    from tracking.z13_readout import HexGeometry,to_eye,to_frame
    from tracking.z13_ridge_localizer import map_center
    from tracking.z13_box_proposal import choose_region,proposal_costs
    p=read(RUN/'protocol.json');verify_sources(p);queries,centers=original.load_queries(read(OLD/'protocol.json'))
    gd=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');geom=HexGeometry(gd['points'],gd['eye_size'])
    maps={};reproduction={};canonical={};hydrated=0
    for split,videos in (('fit',p['fit20']),('evaluate',p['evaluate26'])):
        for a in ('D','C','R1'):
            data,_=load_maps(MAPS/a/split);labels={l['label']:l for l in read(MAPS/a/split/'labels.json')}
            cache={}
            for f in sorted((ROOT/'runs/z13_candidate_reserve_20261021'/split/a).glob('cache_*.npz')):
                with np.load(f) as z:
                    for ep,t,cs,sc in zip(z['episode'],z['t'],z['centers'],z['scores']):cache[int(ep),int(t)]=(cs[0].copy(),float(sc[0]))
            dc=ds=0.
            for v in videos:
                for q in queries[v]:
                    l=q['label'];m=labels[l['label']]
                    assert all(l[k]==m[k] for k in ('video','pos','kind','pred'))
                    if a=='D':l['crop']=m['crop'];hydrated+=1
                    else:assert l['crop']==m['crop']
                    center,score=map_center(data[l['label']][-1],geom)
                    expected,oldscore=cache[l['episode'],l['pos']+4]
                    dc=max(dc,float(np.abs(np.asarray(to_frame(center,l['crop'],gd['eye_size']))-expected).max()));ds=max(ds,abs(score-oldscore))
                    if a=='D':canonical[q['uid']]=(expected,oldscore)
            assert dc<.5 and ds<.001,(split,a,dc,ds)
            maps[split,a]={i:x[-1] for i,x in data.items()};reproduction[split+'_'+a]=dict(rows=len(labels),max_center_px=dc,max_score=ds)
    write(RUN/'reproduction.json',dict(maps=reproduction,original_crop_rows_bound=hydrated))
    tau=read(ROOT/'runs/z13_center_hota_20261021/D/config.json')['policy']['tau']
    decisions={};old_selected={}
    def new_cost_rows(q,original_row,detections,track,get_dists):
        split='fit' if q['split']=='fit20' else 'evaluate';l=q['label'];crop=l['crop']
        eye_boxes=[];support=[]
        for d in detections:
            b=np.asarray(d.xyxy,float);a=to_eye(b[:2],crop,gd['eye_size']);c=to_eye(b[2:],crop,gd['eye_size']);xy=(np.asarray(a)+c)/2
            eye_boxes.append([*a,*c]);support.append(bool(int(d.cls)==int(track.cls) and np.all(xy>=0) and np.all(xy<=gd['eye_size'])))
        eye_boxes=np.asarray(eye_boxes,float).reshape(-1,4);pred=np.asarray(to_eye(l['pred'],crop,gd['eye_size']))
        answer={};selected={}
        for a in ('D','C','R1','geometry'):
            field=maps[split,a][l['label']] if a!='geometry' else -np.sum(((geom.points-pred)/gd['eye_size'])**2,axis=1)
            index,_,valid=choose_region(field,geom.points,eye_boxes,support);selected[a]=dict(index=index,supported=int(valid.sum()))
            for kind in ('center','box'):
                answer[a+'_'+kind]=proposal_costs(track,detections,original_row,get_dists,index,ALPHAS,kind=='box')
        oldcenter,score=canonical[q['uid']];old_selected[q['uid']]=bool(score>=tau)
        oldcost=original_row.copy();saved=track.mean.copy()
        try:
            if score>=tau:track.mean[:2]=oldcenter;oldcost=get_dists([track],detections)[0].copy()
        finally:track.mean[:]=saved
        answer['Dold']=[original_row.copy(),oldcost]
        decisions[q['uid']]=selected
        return answer
    s=source();assert hashlib.sha256(s.encode()).hexdigest()==p['executed_source_sha256']
    ns={**vars(original),'RUN':RUN,'new_cost_rows':new_cost_rows,'T0':time.monotonic()};exec(compile(s,__file__,'exec'),ns)
    times=[];choices=None;hashes={}
    for i,v in enumerate(p['videos']):
        if i==len(p['fit20']):
            fit,_=aggregate(p['fit20']);choices={a:max(range(len(ALPHAS)),key=lambda j:np.mean(list(fit[a][j].values()))) for a in ARMS}
            write(RUN/'choice.json',dict(at=now().isoformat(),indices=choices,alphas={a:ALPHAS[j] for a,j in choices.items()},frozen_before_evaluate26=True,fit_per_video=fit))
        start=time.monotonic();path=RUN/'videos'/(v+'.json');assert not path.exists()
        rows,checks=ns['replay_video'](v,queries[v],centers,p);old={r['uid']:r for r in read(OLD/'videos'/(v+'.json'))['rows']}
        for r in rows:
            if r['kind']>=0:
                out=r['arms']['Dold'][1]
                if old_selected[r['uid']]:assert all(out[k]==old[r['uid']]['arms']['D'][0][k] for k in out)
                else:assert not out['beneficial'] and not out['harmful']
                assert all(not r['arms'][a][0]['beneficial'] and not r['arms'][a][0]['harmful'] for a in ARMS)
            r['proposal_choices']=decisions[r['uid']]
        write(path,dict(rows=rows,checks=checks));hashes[v]=sha(path);times.append(time.monotonic()-start)
        write(RUN/'status.json',dict(state='running',stage='Одна область: перенос центра против полной рамки',variant=v,done=i+1,total=46,updated=now().isoformat(),last_video_seconds=times[-1]))
        if i==1:
            remaining=(datetime.fromisoformat(p['save_at'])-now()).total_seconds();estimate=2*np.mean(times)*44+90
            write(RUN/'price.json',dict(remaining_seconds=remaining,conservative_required_seconds=estimate,passed=bool(estimate<remaining)));assert estimate<remaining,'Original1200s price gate failed'
    assert choices is not None
    allvalues,counts=aggregate(p['evaluate26'],choices)
    values={a:x[choices[a] if a!='Dold' else 0] for a,x in allvalues.items()};comparison=contrasts(values)
    verify_sources(p)
    write(RUN/'report.json',dict(complete=True,primary_pass=primary(comparison),paired=comparison,per_video=values,all_alpha_evaluate=allvalues,counts=counts,video_sha256=hashes,choice_sha256=sha(RUN/'choice.json'),reproduction_sha256=sha(RUN/'reproduction.json'),limits=p['limits']))
    write(RUN/'status.json',dict(state='completed',stage='Полный парный прототип сохранён; не HOTA',done=46,total=46,updated=now().isoformat()))


def finish():
    from scripts.z13_hota_deadline_guard import WindowsProcesses
    p=read(RUN/'protocol.json');e=read(RUN/'controller_result.json');ps=WindowsProcesses()
    assert e['exit_code']==0 and e['all_owned_ended'] and not e['remaining'] and all(ps.creation(int(pid))!=b for pid,b in e['owned'].items())
    verify_sources(p);r=read(RUN/'report.json');choices=read(RUN/'choice.json')['indices']
    for v,h in r['video_sha256'].items():assert sha(RUN/'videos'/(v+'.json'))==h
    values,counts=aggregate(p['evaluate26'],choices);selected={a:v[choices[a] if a!='Dold' else 0] for a,v in values.items()}
    assert values==r['all_alpha_evaluate'] and counts==r['counts'] and selected==r['per_video'] and contrasts(selected)==r['paired'] and primary(r['paired'])==r['primary_pass']
    write(RUN/'independent_audit.json',dict(at=now().isoformat(),all46_baselines_exact=True,all_primary_rows_hashes_choices_and_intervals_verified=True,report_sha256=sha(RUN/'report.json')))
    cat=read(ROOT/'experiments/project_history.json');c=next(c for c in cat['experiments'] if c['id']==IDENT)
    c.update(state='completed_positive' if r['primary_pass'] else 'completed_negative',done=now().strftime('%Y-%m-%d'),summary='Выход полной рамки: '+('критерии пройдены.' if r['primary_pass'] else 'основной критерий не пройден.'),obtained=(RUN/'report.json').relative_to(ROOT).as_posix()+' → paired/counts: '+str(dict(paired=r['paired'],counts=r['counts'])),interpretation='Проверен рабочий выбор по отклику обученной модели цели без разметки на входе. Центр и полная рамка одной выбранной области — парные варианты. Это первые назначения на повторно используемом train26, не HOTA и не предел полезности FlyVis.',decision=p['next'])
    for i,met in enumerate((True,r['primary_pass'])):c['acceptance'][i].update(met=met,evidence=(RUN/('independent_audit.json' if i==0 else 'report.json')).relative_to(ROOT).as_posix())
    write(ROOT/'experiments/project_history.json',cat);write(RUN/'finish_recorded.json',dict(at=now().isoformat(),complete=True,primary_pass=r['primary_pass'],report_sha256=sha(RUN/'report.json')))
    journal('итог рабочего выхода модели цели в полную рамку','*Получено.* '+c['obtained']+'\n\n*Вывод.* '+c['interpretation']+'\n\n*Решение.* '+c['decision']+'\n\nHOTA не измерялась; процент к стенду не определён. Оракульный запас и итог рабочего выбора не смешиваются; относительная доля потолка требует отдельно сопоставимых действий.')
    h=ROOT/'runs/z13_weekend_20261024/HANDOFF.md'
    h.write_text('# Актуально — '+now().strftime('%d.%m.%Y %H:%M')+' MSK\n\n'+IDENT+' окончен, primary_pass='+str(r['primary_pass'])+'. Источник runs/z13_box_proposal_20261025/report.json; выбор alpha только20fit. '+p['next']+' Автор просит итог всех тестов: отдельно геометрический запас и рабочая версия, ничего не выдавать за HOTA. Контроллер завершён, исходники/логи/отрицательные результаты сохранять. Скрытые процессы, одинGPU, до26.10 07:40МСК,57чужихправок/.gitignore сохранять.\n\nНиже прежние передачи сохранены.\n\n'+h.read_text(encoding='utf8'),encoding='utf8')


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--worker' in sys.argv:
        try:worker()
        except Exception as error:write(RUN/'failure.json',dict(at=now().isoformat(),error=repr(error)));raise
    elif '--finish' in sys.argv:finish()
    else:raise RuntimeError('Explicit mode required')
