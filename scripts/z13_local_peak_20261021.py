"""Fixed local peak estimator; CPU-only over identical saved response maps."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,verify,sha,now,MSK
RUN=ROOT/'runs/z13_local_peak_20261021';SOURCE=ROOT/'runs/z13_query_memory_20261021'
IDENT='z13_local_peak_v1';ATTEMPT='full';LIMIT=600;T0=time.monotonic()


def check():
    if time.monotonic()-T0>LIMIT or (RUN/(ATTEMPT+'.stop')).exists() or now()>=datetime(2026,10,21,7,35,tzinfo=MSK):raise TimeoutError('Budget/save deadline')
def status(stage,**kw):
    check();write(RUN/'status.json',dict(state='running',phase=ATTEMPT,stage=stage,updated=now().isoformat(),**kw))
def journal(title,text):
    path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';s=path.read_text(encoding='utf-8');a='## Журнал изменений\n';assert s.count(a)==1
    path.write_text(s.replace(a,a+f'\n**{now():%d.%m.%Y, %H:%M} MSK** — {title} `{IDENT}`\n\n'+text+'\n',1),encoding='utf-8')


def prepare():
    if RUN.exists():raise RuntimeError('Existing pilot must not be duplicated')
    base=read(SOURCE/'protocol.json');verify(base)
    names=['tracking/z13_local_peak.py','scripts/z13_local_peak_20261021.py','tests/test_z13_local_peak.py','runs/z13_query_memory_20261021/protocol.json','tools/flyvis-runtime/flyvis/datasets/rendering/eye.py','runs/flyvis/results/flow/0000/000/_meta.yaml']
    p=dict(id=IDENT,created=now().isoformat(),authorization='Author authorized autonomous necessary runs/repairs/publication until07:40 MSK21.10.2026.',
        action='Replace the weighted center of the same max-column neighborhood by a guarded local quadratic maximum. Old center is already continuous: this is not a claim that subpixel localization was absent.',
        method='Same max and unique up-to7 neighbors. Coordinates divided by lattice spacing. Least-squares quadratic with6 coefficients; accept only rank6, negative Hessian eigmax<-1e-8,condition<=1e6 and vertex in the max-column Voronoi cell; otherwise exactly old center. Confidence and selected row order unchanged. Numerical guards fixed, not tuned.',
        source='Only each original last-observation response map from z13_query_memory cache. Do not combine with chosen query/fixed fusion; same saved center lambda, core and history. Original scalar response confidence unchanged.',
        geometry='Canonical BoxEye extent15/kernel13, actual integer receptor centers and min_frame_size, exactly as EyeBank. CPU initialization only, no network inference; geometry checked by full original-center reproduction.',
        primary='C quadratic-C weighted: mean>=0.02 and paired-video95%lower>0. C quadratic-R1 quadratic lower>0. Difference-in-differences (C local-estimator gain minus R1 gain) lower>0. All required; D secondary, no rescue.',
        secondary='D quadratic-D weighted and D quadratic-R1 quadratic; fit acceptance/fallback reasons; known nonuseful and unknown at unchanged1/2/5% ranking. No parameter search or new training.',
        counterargument='Most reserve may require choosing a different peak; local quadratic shape may be wrong and refinement may worsen localization. This tests a cheap estimator change, not a proven cause of errors.',
        budget_seconds=600,save_at='2026-10-21T07:35:00+03:00',hard_stop='2026-10-21T07:39:30+03:00',
        paper='https://arxiv.org/html/1608.03773 section4.1: grid maximum followed by continuous Newton localization. Our guarded local quadratic on the hex lattice is not C-COT Fourier interpolation.',
        sha256={**base['sha256'],**{n:sha(ROOT/n) for n in names}})
    write(RUN/'protocol.json',p);cp=ROOT/'experiments/project_history.json';cat=read(cp);assert not any(x['id']==IDENT for x in cat['experiments'])
    cat['experiments'].append(dict(id=IDENT,title='З13: локальный квадратичный максимум вместо взвешенного центра',area='flyvis',state='planned',started=f'{now():%Y-%m-%d}',tags=['fly-problem-P8','engineering-future'],
        question='Улучшает ли локальная оценка вершины той же карты центр FlyVis сверх прежнего взвешенного среднего и такого же уточнения на входе глаза?',
        hypothesis='Форма гауссовой цели допускает другую локальную оценку центра; знак выигрыша на реальных картах неизвестен.',
        expected='Один неизменный аналитический узел, одинаковые карты last и confidence, C/D/R1, все train26. Без подбора параметров и обучения. CPU до600с после готовности исходных карт.',
        acceptance=[dict(criterion='Исходный взвешенный центр/confidence точно воспроизведены на8326 строках C/D/R1; хеши и карты сохранены.',met=None),dict(criterion='C quadratic−C weighted: средний сдвиг>=2п.п. и нижняя95%повидеоДИ>0.',met=None),dict(criterion='C quadratic−R1 quadratic и разность улучшений C−R1: обе нижние95%повидеоДИ>0.',met=None)],
        summary='Зафиксирован локальный квадратичный узел; исход не получен.',done='',obtained='',interpretation='',decision='',cannot_claim='Прежний центр уже непрерывный; не доказано, что ошибки вызваны дискретизацией. Не полная C-COT. Нет HOTA, нового выбора пика или безопасного отказа; старый3-пиковый оракул не потолок новых координат.',metrics=[],commits=[],files=names[:2],evidence=[dict(kind='file',ref='runs/z13_local_peak_20261021/protocol.json',claim='Постановка и ограничения до исходов')]))
    write(cp,cat);journal('постановка локального уточнения того же пика','*Постановка.* '+p['action']+'\n\nОпора: '+p['paper']+'\n\n'+p['method']+'\n\nКритерии: '+p['primary']+'\n\nВыбран как короткая проверка следующего узла считывания по неизменным сохранённым картам. Дообучение ядра дороже и не подтверждено предыдущими сравнениями; расширение вырезки потребовало бы других входов. Не утверждается, что найден узел причины ошибок. GPU не нужна; CPU до600с после готовности полного источника, до07:40 МСК.')


def worker():
    import numpy as np
    from tracking.z13_readout import HexGeometry,to_frame
    from tracking.z13_local_peak import quadratic_peak
    from scripts.z13_target_model import center_and_presence
    from scripts.z13_query_memory_20261021 import load_maps
    from scripts.z13_continue import paired,per_video_useful,coverage_table
    p=read(RUN/'protocol.json');verify(p)
    end=read(SOURCE/'full_controller_result.json');assert end['worker_ended'] and end['exit_code']==0 and not end['hard_timeout']
    # Initialize only canonical BoxEye on explicitly selected CPU. Do not
    # invent a regular-hex geometry or a nominal image size.
    import torch
    torch.cuda.is_available=lambda:False
    torch.set_default_device('cpu')
    from tracking.flyvis_motion import activate_flyvis
    flyvis=activate_flyvis()
    assert str(flyvis.device)=='cpu'
    from flyvis.datasets.rendering import BoxEye
    eye=BoxEye(extent=15,kernel_size=13);yx=eye.receptor_centers.cpu().numpy().astype(np.float32)
    size=int(eye.min_frame_size[0]);assert size==int(eye.min_frame_size[1])
    points=np.column_stack([yx[:,1]+size//2,yx[:,0]+size//2]).astype(np.float32)
    geom=HexGeometry(points,size)
    write(RUN/'geometry.json',dict(eye_size=size,points=points.tolist(),spacing=geom.spacing,source='Canonical CPU BoxEye; same integer receptor coordinates as EyeBank'))
    allrows={};manifest={};repro={};fallback={}
    for ai,arm in enumerate(('C','D','R1')):
        status('Локальная вершина тех же карт',variant=arm,done=ai,total=3)
        maps,m=load_maps(SOURCE/arm/'evaluate');manifest.update(m);labels=read(SOURCE/arm/'evaluate/labels.json');old=read(SOURCE/arm/'last_rows.json');old={r['label']:r for r in old};base=[];new=[];reasons=Counter();dc=ds=0.
        for l in labels:
            check();score=maps[l['label']][-1];c,pres=center_and_presence(score,geom);q,p2,reason=quadratic_peak(score,geom);assert pres==p2;reasons[reason]+=1
            pair=[]
            for center in (c,q):
                ce=np.asarray(to_frame(center,l['crop'],size));r={k:l[k] for k in ('label','video','kind','pos','pred')};r.update(center=ce.tolist(),p_present=pres)
                if l['kind']==1:
                    err=float(np.linalg.norm(ce-l['gt']));ep=float(np.linalg.norm(np.asarray(l['pred'])-l['gt']));r.update(err=err,err_pred=ep,scale=l['scale'],useful=bool(err<=.25*l['scale'] and err<ep))
                elif l['kind']==0:r['useful']=False
                pair.append(r)
            base.append(pair[0]);new.append(pair[1]);saved=old[l['label']];dc=max(dc,float(np.abs(np.asarray(pair[0]['center'])-saved['center']).max()));ds=max(ds,abs(pres-saved['p_present']))
        assert len(base)==8326 and dc<1e-6 and ds<1e-9
        repro[arm]=dict(rows=len(base),max_center_px=dc,max_score=ds,confidence_exactly_unchanged=True);fallback[arm]=dict(reasons)
        for mode,rows in [('weighted',base),('quadratic',new)]:allrows[arm+'_'+mode]=rows;write(RUN/arm/(mode+'_rows.json'),rows)
    pv={k:per_video_useful(v) for k,v in allrows.items()};comp={}
    for arm in ('C','D'):
        comp[arm+'_quadratic-'+arm+'_weighted']=paired(pv[arm+'_quadratic'],pv[arm+'_weighted']);comp[arm+'_quadratic-R1_quadratic']=paired(pv[arm+'_quadratic'],pv['R1_quadratic'])
    cg={v:pv['C_quadratic'][v]-pv['C_weighted'][v] for v in pv['C_weighted']};rg={v:pv['R1_quadratic'][v]-pv['R1_weighted'][v] for v in pv['R1_weighted']};comp['C_gain-R1_gain']=paired(cg,rg)
    c=comp['C_quadratic-C_weighted'];passed=c['mean_per_video']>=.02 and c['ci95'][0]>0 and comp['C_quadratic-R1_quadratic']['ci95'][0]>0 and comp['C_gain-R1_gain']['ci95'][0]>0
    verify(p);write(RUN/'reproduction.json',repro);write(RUN/'cache_manifest.json',manifest);write(RUN/'report.json',dict(primary_pass=passed,paired=comp,fallback=fallback,summary={k:dict(rows=len(v),useful_mean_video=float(np.mean(list(pv[k].values()))),coverage=coverage_table(v)) for k,v in allrows.items()},limitations='Old center already continuous; same peak and confidence. Exploratory train26, no HOTA or safe refusal.'))


def main():
    global ATTEMPT,LIMIT
    a=argparse.ArgumentParser();a.add_argument('--prepare',action='store_true');a.add_argument('--worker',action='store_true');a.add_argument('--attempt',default='full');a.add_argument('--budget',type=int,default=600);x=a.parse_args()
    if x.prepare:prepare();return
    ATTEMPT=x.attempt;LIMIT=x.budget;os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
    if not x.worker:raise ValueError('Choose prepare/worker')
    try:worker();write(RUN/'status.json',dict(state='completed',phase=ATTEMPT,stage='Локальное уточнение сохранено',updated=now().isoformat()))
    except Exception as exc:write(RUN/'status.json',dict(state='failed',phase=ATTEMPT,stage=str(exc),updated=now().isoformat()));raise


if __name__=='__main__':main()
