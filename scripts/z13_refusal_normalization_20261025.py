"""Third bounded diagnosis: remove only score-map scale normalization."""
from pathlib import Path
from collections import Counter
from datetime import timedelta
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.z13_refusal_causes_20261025 import RUN as BASE,OLD,IDENT,ARMS,read,write,sha,now,journal
RUN=BASE/'normalization'
MAPS=ROOT/'runs/z13_query_memory_20261021'


def prepare():
    assert not RUN.exists()
    assert read(BASE/'finish_recorded.json')['all_owned_ended']
    names=[Path(__file__).relative_to(ROOT).as_posix(),'scripts/z13_refusal_trace_console.ps1',
        'scripts/z13_bounded_stage_20261022.py','scripts/z13_query_memory_20261021.py',
        'tracking/z13_ridge_localizer.py','tracking/z13_readout.py','scripts/z13_continue.py',
        'runs/z13_local_peak_20261021/geometry.json',(BASE/'protocol.json').relative_to(ROOT).as_posix(),
        (BASE/'report.json').relative_to(ROOT).as_posix(),(BASE/'rows.json').relative_to(ROOT).as_posix(),
        (BASE/'diagnosis.json').relative_to(ROOT).as_posix(),
        'runs/z13_absolute_confidence_20261022/report.json']
    for arm in ARMS:
        names.append(f'runs/z13_target_center_20261018/{arm}/center_rows.json')
        names.append((MAPS/arm/'evaluate/labels.json').relative_to(ROOT).as_posix())
        names += [x.relative_to(ROOT).as_posix() for x in (MAPS/arm/'evaluate').glob('maps_*.*')]
    p=dict(id=IDENT,device='cpu',budget_seconds=600,created=now().isoformat(),
        save_at=(now()+timedelta(seconds=600)).isoformat(),hard_stop=(now()+timedelta(seconds=620)).isoformat(),
        authorization='Third and last bounded diagnosis for the current refusal action, explicitly requested causes plus standing weekend authorization.',
        question='Does dividing the centered peak by map standard deviation cause harmful selection, beyond the already failed raw-max center-utility test?',
        action='Keep every cached canonical top1 center and all one-step outcomes fixed. Replace only confidence by centered peak=max(map)-mean(map), removing only division by standard deviation. Absolute max is a previously tested control, now labelled by assignment outcome without recomputing FlyVis.',
        method='C/D/R1,all8326saved rows. One fixed global top84 including unknown cases,stable source order for ties; verify z-score chosen keys reproduce current tau policy exactly. Raw map centers<0.5px,z score<0.001 from canonical source. New scores do not read labels. Report selected known useful/harmful/neutral,unknown separately; paired video bootstrap5000 seed1 on net beneficial-minus-harmful frequency over known queries. Reusedtrain26,not a deployed policy or independent confirmation.',
        criterion='Denominator-removal selector is worth separate implementation only if both paired95%lower(D_centered_peak-D_z)>0 and paired95%lower(D_centered_peak-R1_centered_peak)>0. Otherwise do not implement/promote this replacement. This is a scalar-scoring ablation,not the previously blocked supervised assignment-label training; its2%reserve gate remains unchanged.',
        counterargument='Removing standard deviation may recover intensity sensitivity but still fail to identify the target or utility relative to the tracker; map magnitude may vary by exemplar. A changed selection is not causal evidence of better representations or higher HOTA.',
        limits='No new training,feature extraction,threshold search,dev/20val tuning,GPU,HOTA or75videos. A negative result rejects only this scalar replacement,not the entire refusal problem. After this third diagnosis choose one concrete node repair or record why it cannot yet be justified; no endless ablation sweep.',
        sha256={n:sha(ROOT/n) for n in sorted(set(names))})
    write(RUN/'protocol.json',p);write(RUN/'status.json',dict(state='ready',stage='Роль нормировки в отказе,без изменения центров',done=0,total=3,updated=now().isoformat()))
    journal('третья и последняя проверка текущего разбора:знаменатель уверенности',
        '*Постановка.* '+p['question']+'\n\n*Действие.* '+p['action']+'\n\n*Критерий до исходов.* '+p['criterion']+
        '\n\n*Контраргумент.* '+p['counterargument']+'\n\n'+p['method']+'\n\n'+p['limits']+
        '\n\nЧисло62из84отрицательныхужеизвестно22.10поabsolute_confidence/report.json;новизнаэтойпроверкинеприписываетсяэтомуфакту. '
        'Полныестарые карты найдены вz13_query_memory/evaluate;повторногоизвлечениянет. До600сCPU,свойконтроллерисрок.')


def worker():
    from scripts.z13_query_memory_20261021 import load_maps
    from tracking.z13_readout import HexGeometry,to_frame
    from tracking.z13_ridge_localizer import map_center
    from scripts.z13_continue import paired
    from scripts.z13_localization_labels_20261023 import verify_sources
    p=read(RUN/'protocol.json');verify_sources(p)
    geomdata=read(ROOT/'runs/z13_local_peak_20261021/geometry.json')
    geom=HexGeometry(np.asarray(geomdata['points'],np.float32),geomdata['eye_size'])
    bound=read(BASE/'rows.json');allresults={};stats={};choices={};repro={}
    for ai,arm in enumerate(ARMS):
        if (RUN/'stop.request').exists() or now().isoformat()>=p['save_at']:
            raise TimeoutError('Normalization diagnosis deadline')
        write(RUN/'status.json',dict(state='running',stage='Контроль карты и замена только деления на разброс',variant=arm,done=ai,total=3,updated=now().isoformat()))
        maps,_=load_maps(MAPS/arm/'evaluate');labels=read(MAPS/arm/'evaluate/labels.json')
        old={r['label']:r for r in read(ROOT/f'runs/z13_target_center_20261018/{arm}/center_rows.json')}
        outcome={r['label']:r for r in bound if r['arm']==arm}
        assert len(maps)==len(labels)==len(old)==len(outcome)==8326
        values={k:[] for k in ('z','centered_peak','absolute_peak')};mapstats=[];dc=ds=0.
        for label in labels:
            idx=label['label'];row=old[idx];score=np.asarray(maps[idx][-1],float)
            center,z=map_center(score,geom);point=np.asarray(to_frame(center,label['crop'],geomdata['eye_size']))
            dc=max(dc,float(np.max(np.abs(point-row['center']))));ds=max(ds,abs(z-row['p_present']))
            assert row['video']==label['video'] and row['kind']==label['kind']
            values['z'].append(row['p_present']);values['centered_peak'].append(float(score.max()-score.mean()));values['absolute_peak'].append(float(score.max()))
            mapstats.append(dict(label=idx,kind=row['kind'],mean=float(score.mean()),sd=float(score.std()),peak=float(score.max()),z=row['p_present']))
        assert dc<.5 and ds<.001,(arm,dc,ds)
        ids=[l['label'] for l in labels];repro[arm]=dict(rows=8326,max_center_px=dc,max_z_score=ds)
        stats[arm]={};allresults[arm]={};choices[arm]={}
        for mode,scores in values.items():
            ordered=sorted(range(len(ids)),key=lambda i:-scores[i]);selected={ids[i] for i in ordered[:84]}
            if mode=='z':assert selected=={idx for idx,r in outcome.items() if r['selected']}
            count=Counter();pv={v:dict(known=0,beneficial=0,harmful=0) for v in sorted({r['video'] for r in outcome.values()})}
            for idx,r in outcome.items():
                if r['kind']>=0:pv[r['video']]['known']+=1
                if idx not in selected:continue
                count['selected']+=1;count['unknown' if r['kind']<0 else 'known']+=1
                count[r['outcome']]+=1 if r['outcome']!='unknown' else 0
                if r['kind']>=0:
                    pv[r['video']]['beneficial']+=r['outcome']=='beneficial';pv[r['video']]['harmful']+=r['outcome']=='harmful'
            net={v:(x['beneficial']-x['harmful'])/x['known'] for v,x in pv.items()}
            allresults[arm][mode]=dict(counts=dict(count),per_video=pv,net_frequency=net)
            choices[arm][mode]=sorted(selected)
        for kind in (0,1,-1):
            subset=[s for s in mapstats if s['kind']==kind]
            stats[arm][str(kind)]={name:np.quantile([s[name] for s in subset],[.1,.5,.9]).tolist() for name in ('sd','peak','z')}
        write(RUN/(arm+'_map_statistics.json'),mapstats)
        del maps
    contrasts={}
    for arm in ARMS:
        for mode in ('centered_peak','absolute_peak'):
            contrasts[arm+'_'+mode+'-'+arm+'_z']=paired(allresults[arm][mode]['net_frequency'],allresults[arm]['z']['net_frequency'])
    contrasts['D_centered_peak-R1_centered_peak']=paired(allresults['D']['centered_peak']['net_frequency'],allresults['R1']['centered_peak']['net_frequency'])
    passed=contrasts['D_centered_peak-D_z']['ci95'][0]>0 and contrasts['D_centered_peak-R1_centered_peak']['ci95'][0]>0
    verify_sources(p)
    write(RUN/'report.json',dict(complete=True,primary_pass=bool(passed),reproduction=repro,results=allresults,paired=contrasts,chosen_labels=choices,statistics=stats,not_hota=True,limits=p['limits']))
    write(RUN/'status.json',dict(state='completed',stage='Контроль нормировки завершён;не HOTA',done=3,total=3,updated=now().isoformat()))


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--worker' in sys.argv:
        try:worker()
        except Exception as error:
            write(RUN/'status.json',dict(state='failed',stage=repr(error),updated=now().isoformat()));raise
    else:raise ValueError('Choose --prepare or --worker')
