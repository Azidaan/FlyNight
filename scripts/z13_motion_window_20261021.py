"""One bounded map-localization replacement using a motion-centered cosine window."""
from __future__ import annotations
import argparse
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,verify,now,MSK
RUN=ROOT/'runs/z13_motion_window_20261021';SOURCE=ROOT/'runs/z13_query_memory_20261021';BCE=ROOT/'runs/z13_candidate_data104_20261021'
IDENT='z13_motion_window_v1';ATTEMPT='full';LIMIT=600;T0=time.monotonic()


def check():
    if time.monotonic()-T0>LIMIT or (RUN/(ATTEMPT+'.stop')).exists() or now()>=datetime(2026,10,21,7,35,tzinfo=MSK):raise TimeoutError('Budget/save deadline')
def status(stage,**kw):
    check();write(RUN/'status.json',dict(state='running',phase=ATTEMPT,stage=stage,updated=now().isoformat(),elapsed_seconds=int(time.monotonic()-T0),**kw))
def journal(title,text):
    path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';s=path.read_text(encoding='utf-8');a='## Журнал изменений\n';assert s.count(a)==1
    path.write_text(s.replace(a,a+f'\n**{now():%d.%m.%Y, %H:%M} MSK** — {title} `{IDENT}`\n\n'+text+'\n',1),encoding='utf-8')
def prepare():
    if RUN.exists():raise RuntimeError('Existing run; inspect without duplication')
    base=read(SOURCE/'protocol.json');verify(base)
    assert not read(ROOT/'runs/z13_pairwise_map_20261021/report_train26.json')['primary_pass']
    names=['tracking/z13_motion_window.py','tests/test_z13_motion_window.py','scripts/z13_motion_window_20261021.py','runs/z13_local_peak_20261021/geometry.json','runs/z13_map_transfer104_20261021/report.json']
    for arm in ('C','D','R1'):
        names.extend(f'runs/z13_query_memory_20261021/{arm}/{f}' for f in ('cache_manifest.json','last_rows.json','fit/labels.json','evaluate/labels.json'))
        names.extend([f'runs/z13_candidate_identity_20261021/{arm}/kalman_rows.json',f'runs/z13_candidate_data104_20261021/{arm}/geometry/ensemble_rows.json'])
    p=dict(id=IDENT,created=now().isoformat(),authorization='Explicit autonomous necessary runs/repairs/publication until07:40 MSK21.10.2026.',
        action='Replace unrestricted map maximum by a forecast-centered cosine-window preference over all721columns inside the FlyVis readout. Keep original local7-neighbor center weighting and raw peak confidence unchanged. This is not confidence filtering or core training.',
        method='Response density=(map-min)/sum(map-min), uniform if flat. Hann window: product overx,y of0.5*(1+cos(2*pi*clip((point-forecast_eye)/eye_size,-0.5,0.5))), normalized to sum1. If window has no support use response itself. Blend densities (1-a)*response+a*window for selecting a column only. Original raw-map local weights around the selected column; a=0 exactly old center/confidence.',
        fit='Only previous20 training videos; fixed influences[0,.125,.25,.5,.75,1]. Select highest per-video useful mean per arm, first grid choice on exact ties. No new seed, core, normalization, ridge lambda, candidate feature fitting or evaluation26 tuning.',
        primary='C_window-C_top1: mean>=0.02 and paired-video95%lower>0. C_window-C_nearest3, C_window-C_BCE104 and C_window-R1_window: all lower>0. All conditions required; D secondary cannot rescue. R1 effect and Cgain-R1gain always reported.',
        counterargument='A spatial window can merely reproduce motion prediction or prefer tiny apparent corrections, without recognizing the target. Full-column choices differ from the old3-peak set. Nearest3 and the existing BCE104 map-reading node are required controls; pixel/relative errors are reported, not only useful fraction.',
        decision='If train26 gate fails, stop this window variant without dev. If it passes, freeze influence values and separately register conditional fixed dev35 full-map extraction under the remaining original morning window before any dev calculation. No dev retuning or protected75.',
        reference=dict(url='https://arxiv.org/html/1606.09549v2',section='2.4 Tracking algorithm',supports='SiamFC adds a cosine window to penalize large displacements. Here center is the current Kalman forecast, search crop is unchanged2x, and local weighting remains ours; no SiamFC reproduction or transferred performance claim.'),
        budget_seconds=600,save_at='2026-10-21T07:35:00+03:00',hard_stop='2026-10-21T07:39:30+03:00',prohibited='No GPU extraction,core training,downloads,dev,HOTA or protected75 in this step.',
        sha256={**base['sha256'],**{n:sha(ROOT/n) for n in names}})
    write(RUN/'protocol.json',p);cp=ROOT/'experiments/project_history.json';cat=read(cp);assert not any(c['id']==IDENT for c in cat['experiments'])
    cat['experiments'].append(dict(id=IDENT,title='З13: окно прогноза внутри чтения полной карты',area='flyvis',state='ready',started=f'{now():%Y-%m-%d}',tags=['fly-problem-P8','engineering-future'],question='Поможет ли мягкий учёт прогноза при выборе позиции полной карты сверх максимума, ближайшего пика и обученного выбора карты?',hypothesis='Совместный ответ внешнего вида и пространственного ожидания может выбрать более полезную позицию; преимущество над одной геометрией не установлено.',expected='Одно окно,6 заранее заданных весов только на20fit, одинаковые C/D/R1 и все train26. Те же сохранённые полные карты, без вычислений ядра; CPU до600с.',
        acceptance=[dict(criterion='Исходный центр/балл всех8326 строк C/D/R1 воспроизведён при нулевом весе: центр<1e-6px,балл<1e-9; исходные карты/хеши неизменны.',met=None),dict(criterion=p['primary'],met=None)],summary='Зарегистрирована одна версия считывания полной карты с пространственным окном; исход не получен.',done='',obtained='',interpretation='',decision='',cannot_claim='Не установлено, что ошибки вызваны отсутствием окна. Смещение к прогнозу не равно узнаванию цели. Старый3-пиковый оракул не потолок новых координат. Нет HOTA, безопасного отказа, новых свойств ядра или независимого подтверждения.',metrics=[],commits=[],files=['tracking/z13_motion_window.py','scripts/z13_motion_window_20261021.py'],evidence=[dict(kind='file',ref='runs/z13_motion_window_20261021/protocol.json',claim='Механизм/контроли/критерии до исходов')]))
    write(cp,cat);journal('постановка окна прогноза внутри считывания','*Постановка.* '+p['action']+'\n\n'+p['method']+'\n\n'+p['fit']+'\n\nКритерии до исходов: '+p['primary']+'\n\nОпора: Bertinetto et al.2016, раздел2.4, https://arxiv.org/html/1606.09549v2 — косинусное окно на карте; наша версия использует прогноз Калмана и прежнюю2x вырезку, это не воспроизведение SiamFC. Главный контраргумент: '+p['counterargument']+'\n\n'+p['decision']+' CPU до600с; нового прохода ядра не требуется.')


def make_rows(maps,labels,geom,eye_size,influence):
    import numpy as np
    from tracking.z13_motion_window import motion_window_center
    from tracking.z13_readout import to_eye,to_frame
    out=[]
    for l in labels:
        check();score=maps[l['label']][-1];pred_eye=to_eye(l['pred'],l['crop'],eye_size);ce,pres=motion_window_center(score,geom,pred_eye,eye_size,influence);center=np.asarray(to_frame(ce,l['crop'],eye_size));r={k:l[k] for k in ('label','video','kind','pos','pred')};r.update(center=center.tolist(),p_present=pres)
        if l['kind']==1:
            err=float(np.linalg.norm(center-l['gt']));ep=float(np.linalg.norm(np.asarray(l['pred'])-l['gt']));r.update(err=err,err_pred=ep,scale=l['scale'],useful=bool(err<=.25*l['scale'] and err<ep))
        elif l['kind']==0:r['useful']=False
        out.append(r)
    return out


def worker():
    import numpy as np
    from tracking.z13_readout import HexGeometry
    from scripts.z13_query_memory_20261021 import load_maps
    from scripts.z13_continue import paired,per_video_useful,coverage_table
    p=read(RUN/'protocol.json');verify(p);g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');geom=HexGeometry(np.asarray(g['points'],np.float32),g['eye_size']);results={};choices={};manifest={};repro={}
    for ai,arm in enumerate(('C','D','R1')):
        maps,m=load_maps(SOURCE/arm/'fit');manifest.update(m);labels=read(SOURCE/arm/'fit/labels.json');grid=[]
        for j,a in enumerate((0.,.125,.25,.5,.75,1.)):
            status('Вес окна только по20fit',variant=arm,done=j,total=6,arms_done=ai,arms_total=3);rr=make_rows(maps,labels,geom,g['eye_size'],a);grid.append(dict(influence=a,fit_mean_video=float(np.mean(list(per_video_useful(rr).values())))))
        best=max(grid,key=lambda r:r['fit_mean_video']);choices[arm]=dict(best=best,grid=grid);write(RUN/arm/'choice.json',choices[arm]);del maps
        maps,m=load_maps(SOURCE/arm/'evaluate');manifest.update(m);labels=read(SOURCE/arm/'evaluate/labels.json');status('Полная карта на26 без подбора',variant=arm,done=0,total=2,arms_done=ai,arms_total=3)
        top=make_rows(maps,labels,geom,g['eye_size'],0.);current=make_rows(maps,labels,geom,g['eye_size'],best['influence']);old={r['label']:r for r in read(SOURCE/arm/'last_rows.json')};assert len(top)==len(old)==8326
        dc=max(float(np.abs(np.asarray(r['center'])-old[r['label']]['center']).max()) for r in top);ds=max(abs(r['p_present']-old[r['label']]['p_present']) for r in top);assert dc<1e-6 and ds<1e-9;repro[arm]=dict(rows=len(top),max_center_px=dc,max_score=ds)
        for mode,rows in [('top1',top),('window',current)]:results[arm+'_'+mode]=rows;write(RUN/arm/(mode+'_rows.json'),rows)
        results[arm+'_nearest']=read(ROOT/f'runs/z13_candidate_identity_20261021/{arm}/kalman_rows.json');results[arm+'_BCE']=read(BCE/arm/'geometry/ensemble_rows.json');expected=read(SOURCE/arm/'cache_manifest.json');assert all(expected[k]==v for k,v in m.items());del maps
    pv={k:per_video_useful(v) for k,v in results.items()};comp={}
    for arm in ('C','D'):
        for base in (arm+'_top1',arm+'_nearest',arm+'_BCE','R1_window'):comp[arm+'_window-'+base]=paired(pv[arm+'_window'],pv[base])
    comp['R1_window-R1_top1']=paired(pv['R1_window'],pv['R1_top1']);a={v:pv['C_window'][v]-pv['C_top1'][v] for v in pv['C_top1']};b={v:pv['R1_window'][v]-pv['R1_top1'][v] for v in pv['R1_top1']};comp['C_gain-R1_gain']=paired(a,b)
    d=comp['C_window-C_top1'];passed=d['mean_per_video']>=.02 and d['ci95'][0]>0 and all(comp[k]['ci95'][0]>0 for k in ('C_window-C_nearest','C_window-C_BCE','C_window-R1_window'))
    verify(p);write(RUN/'cache_manifest.json',manifest);write(RUN/'reproduction.json',repro)
    write(RUN/'report.json',dict(primary_pass=passed,paired=comp,choices=choices,summary={k:dict(rows=len(v),useful_mean_video=float(np.mean(list(pv[k].values()))),median_error_relative=float(np.median([r['err']/r['scale'] for r in v if r['kind']==1])),coverage=coverage_table(v)) for k,v in results.items()},limitations='Same full maps/history; current forecast only, fit20 choices fixed. Not SiamFC reproduction, no HOTA; old3-peak oracle is not this new-coordinate ceiling.'))


def main():
    global ATTEMPT,LIMIT
    a=argparse.ArgumentParser();a.add_argument('--prepare',action='store_true');a.add_argument('--worker',action='store_true');a.add_argument('--attempt',default='full');a.add_argument('--budget',type=int,default=600);x=a.parse_args()
    if x.prepare:prepare();return
    if not x.worker:raise ValueError('Choose action')
    ATTEMPT=x.attempt;LIMIT=x.budget;os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
    try:worker();write(RUN/'status.json',dict(state='completed',phase=ATTEMPT,stage='Чтение карты с окном сохранено',updated=now().isoformat()))
    except Exception as exc:write(RUN/'status.json',dict(state='failed',phase=ATTEMPT,stage=str(exc),updated=now().isoformat()));raise


if __name__=='__main__':main()
