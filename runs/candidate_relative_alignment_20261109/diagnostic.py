"""Exploratory coordinate-only diagnostic; no new model, detector or ID action."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import sys,json,hashlib
from pathlib import Path
from datetime import datetime,timedelta,timezone
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;BASE=ROOT/'runs/historical_two_candidate_comparison_20261109'
sys.path.insert(0,str(BASE));import study_v2 as s
import numpy as np
from tracking.observed_spatial_reader import ObservedSpatialReader
from tracking.z13_readout import HexGeometry
from scripts.z13_target_model import neighbor_aug
read=s.read;sha=s.sha;cast=s.cast
def write(n,v):
    with (OUT/n).open('x',encoding='utf8') as f:json.dump(cast(v),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def math():
    # Positive diagonal affine transform is invertible. A linear field is exact
    # under a linear interpolator; correlation alone cannot identify a ramp.
    from scipy.spatial import Delaunay
    from tracking.hex_linear_interpolation import sample_linear
    from tracking.channel_balanced_correspondence import channel_balanced_similarity
    rng=np.random.default_rng(42);x=rng.uniform(0,1,(100,2));a=np.array([10.,20.]);b=np.array([30.,40.]);scale=np.array([2.,.7]);y=b+(x-a)*scale;back=a+(y-b)/scale
    roundtrip=float(np.max(abs(x-back)));assert roundtrip<1e-12
    points=np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.],[.4,.6]]);mesh=Delaunay(points);q=rng.uniform(.05,.95,(100,2));field=points@np.array([2.,3.])+4.;pred=sample_linear(mesh,field[:,None],q)[:,0];err=float(np.max(abs(pred-(q@np.array([2.,3.])+4.))));assert err<1e-12
    ramp=np.array([0.,1.,2.,3.])[:,None];rho=channel_balanced_similarity(ramp,2*ramp+7);assert abs(rho-1)<1e-12
    write('math.json',dict(passed=True,inverse_max_error=roundtrip,affine_field_max_error=err,wrong_identity_ramp_can_score=rho,conclusion='Coordinate transform is mathematically defined and interpolation reproduces affine fields. This is not an identity proof. Centered ramp correlation can be one for a wrong object; occlusion can invalidate box-part equivalence.'))
def register():
    assert read(OUT/'math.json')['passed'];p=read(BASE/'protocol.json');s.verify(p);s.verify(read(BASE/'maps_complete.json'));assert not read(BASE/'summary.json')['reader_gate_pass']
    files=[Path(__file__),OUT/'math.json',BASE/'protocol.json',BASE/'execution_protocol.json',BASE/'results.json',BASE/'summary.json',BASE/'maps_complete.json',ROOT/'tracking/box_normalized_correspondence.py',ROOT/'tracking/own_template_verification.py',ROOT/'tracking/observed_spatial_reader.py',ROOT/'tracking/hex_linear_interpolation.py',ROOT/'tracking/channel_balanced_correspondence.py']
    write('protocol.json',dict(registered=s.stamp(),sources={f.relative_to(ROOT).as_posix():sha(f) for f in files},cases=[c['uid'] for c in p['cases']],scope='Exploratory diagnosis on already evaluated twelve queries, not independent validation. No model replay, threshold tuning, new candidate or ID write.',formula='T_k(x)=Bk_lo+(x-B0_lo)*(Bk_size/B0_size), B0=archived reference STATE box, Bk=current RAW observed box. Same nineteen physical-pixel offsets added AFTER transform. Positive box extents required. No scale clipping, rotation/anchor optimization or mask.',support='Per candidate intersect original full-19 support with affine full-19 box/hull support. Compare prediction and affine on EXACT SAME reference cell indices for that candidate. Never intersect two candidates. Minimum three unchanged; abstention retained in denominator. Ready if at least eight queries support BOTH candidates before evaluating any new scores.',controls='All seven previous representations and original local cost. Reuse original grids if common support identical. Primary descriptive comparison C34 affine_common versus prediction_common; original_full is context, not a support-matched causal comparison.',history='Observed-box relative correspondence is NOT new: tracking/own_template_verification.py already tested it with nearest cell matching, global centered cosine and old C8/R1/image/DIS. Here only continuous interpolation, frozen channel-balanced reader/full34 and fixed19 shifts form a different combination. Lewis NCC source confirms scale/perspective limitations, not this exact engineering hypothesis.',literature='https://scribblethink.org/Work/nvisionInterface/nip.html',acceptance=[dict(criterion='Math, all source hashes and identical per-candidate common support verified; no new map or ID write.',met=None),dict(criterion='At least eight queries support both candidates on frozen common support before scoring.',met=None)],stop='If readiness fails record geometry only and do not loosen cells/offsets. If scores do not improve do not retune. A positive descriptive result requires a distinct frozen test before deployment.'))
    path=ROOT/'experiments/project_history.json';cat=read(path);assert not any(c['id']==OUT.name for c in cat['experiments']);cat['experiments'].append(dict(id=OUT.name,title='Привязка карты к конкретной рамке — диагностика координат на готовых картах',area='flyvis',state='running',started='2026-11-09',question='Меняется ли выбор при сопоставлении относительных участков каждой рамки, если число и номера опорных ячеек неизменны?',hypothesis='Неправильный порядок может объясняться общим прогнозом положения, а не недостаточностью сигнала. Обратная версия: масштабирование по рамке само нарушает соответствие частей машины.',expected='Одна ограниченная проверка семи представлений без новой сети, обучения и порогов. Прежний относительный метод учтён; независимым переносом эти данные не являются.',acceptance=read(OUT/'protocol.json')['acceptance'],evidence=[dict(kind='file',ref=f'runs/{OUT.name}/protocol.json',claim='Математика и правила зафиксированы до новых оценок')]))
    path.write_text(json.dumps(cat,ensure_ascii=False,indent=2)+'\n',encoding='utf8');s.journal('Проверено математическое основание привязки карты к рамке',f'*Постановка (`{OUT.name}`).* Проверена обратимость положительного покоординатного преобразования и воспроизведение линейного поля; одновременно показано, что корреляция линейных профилей не доказывает личность. На тех же 12 уже оценённых запросах допускается только исследовательская диагностика координат. Одинаковая поддержка для двух вариантов внутри каждого кандидата, прежние 19 сдвигов и семь представлений. Сначала геометрический допуск 8 запросов с двумя поддержанными кандидатами, затем оценки. Относительные координаты ранее уже проверялись другим считывателем; новой эта идея не объявляется. Источники `runs/{OUT.name}/math.json`, `protocol.json`.')
def run():
    p=read(OUT/'protocol.json');s.verify(p);bp=read(BASE/'protocol.json');s.verify(bp);s.verify(read(BASE/'maps_complete.json'));g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');reader=ObservedSpatialReader(g['points'],g['eye_size'],[1280,720]);nb=HexGeometry(np.array(g['points']),g['eye_size']).neighbors;plans=[];ready=0
    for c in bp['cases']:
        rc=np.array(c['sequence'][c['reference_relative']][1]);ct=np.array(c['sequence'][-1][1]);rs=(rc[2:]-rc[:2])/g['eye_size'];cs=(ct[2:]-ct[:2])/g['eye_size'];old=np.array(c['reference_raw']['state_box']);x=reader.points*rs+rc[:2];rows=[]
        for k in c['candidates']:
            box=np.array(k['box']);assert min(*(box[2:]-box[:2]),*(old[2:]-old[:2]))>0;scale=(box[2:]-box[:2])/(old[2:]-old[:2]);off=np.array(k['plan']['offsets']);trial=box[:2]+(x-old[:2])*scale;trial=trial[:,None]+off;query=(trial-ct[:2])/cs
            keep=((trial>=box[:2])&(trial<=box[2:])).all((1,2))&(reader.mesh.find_simplex(query)>=0).all(1);idx=np.array(k['plan']['cells']);mask=keep[idx];common=idx[mask];oldq=np.array(k['plan']['query'])[mask];newq=query[common]
            assert len(common)==len(oldq)==len(newq);rows.append(dict(kind=k['kind'],cells=common,original_cells=len(idx),scale=scale,prediction=dict(cells=common,query=oldq),relative=dict(cells=common,query=newq),old_scores_reusable=bool(mask.all())))
        supported=all(len(k['cells'])>=3 for k in rows);ready+=supported;plans.append(dict(uid=c['uid'],video=c['video'],supported=supported,candidates=rows))
    write('readiness.json',dict(passed=ready>=8,supported_queries=ready,total=12,plans=plans));print('READY',ready,12,flush=True)
    if ready<8:return
    original=read(BASE/'results.json');byuid={r['uid']:r for r in original['records']};records=[];new=0;reused=0
    def features(key):
        a=s.arrays(key,bp);c=a['C34'].astype(float);r=a['R1'].astype(float);cen=r[:,s.SELECT[:4]]
        return dict(C34=c,FlyVis=c[:,4:12],R134_temporal=r,R134_spatial=neighbor_aug(r[:,:17].T[None],nb)[0].T,input8_temporal=r[:,s.SELECT],center8=np.concatenate([cen,cen],1),neighbor8=neighbor_aug(cen.T[None],nb)[0].T)
    for c,plan in zip(bp['cases'],plans):
        ref,cur=[features(k) for k in c['keys']];record=dict(uid=c['uid'],video=c['video'],supported=plan['supported'],variants={})
        for name in s.NAMES:
            variants=[]
            for k in plan['candidates']:
                old=next(k0 for k0 in byuid[c['uid']]['variants'][name] if k0['kind']==k['kind']);vals={}
                for mode in ['prediction','relative']:
                    if mode=='prediction' and k['old_scores_reusable']:q=dict(scores=old['scores'],score=old['maximum'],winner=old['winner']);reused+=1
                    else:q=reader.compare(ref[name],cur[name],k[mode]);new+=len(q['scores']) if q['scores'] is not None else 0
                    vals[mode]=dict(fixed=q['scores'][9] if q['scores'] is not None else None,maximum=q['score'],winner=q['winner'],scores=q['scores'])
                variants.append(dict(kind=k['kind'],cells=len(k['cells']),values=vals))
            record['variants'][name]=variants
        records.append(record)
    s.verify(p);write('results.json',dict(passed=True,records=records,new_scalar_scores=new,reused_original_grids=reused,no_new_maps=True,no_threshold_tuning=True))
if __name__=='__main__':{'math':math,'register':register,'run':run}[sys.argv[1]]()
