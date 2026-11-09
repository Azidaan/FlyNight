"""Whole relative-template replication with the frozen joint reader."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;BASE=ROOT/'runs/historical_two_candidate_comparison_20261109';OLD=ROOT/'runs/candidate_relative_alignment_20261109'
sys.path.insert(0,str(BASE));import study_v2 as s
import numpy as np
from tracking.observed_spatial_reader import ObservedSpatialReader
from tracking.z13_readout import HexGeometry
from scripts.z13_target_model import neighbor_aug
read=s.read;sha=s.sha
def write(n,v):
    with (OUT/n).open('x',encoding='utf8') as f:json.dump(s.cast(v),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def math():
    assert read(OLD/'math.json')['passed'];assert not read(OLD/'readiness.json')['passed']
    x=np.array([[.2,.3],[.7,.8],[.4,.6]]);old=np.array([0.,0.,1.,1.]);boxes=[np.array([2.,3.,4.,5.]),np.array([6.,3.,8.,5.])];q=[b[:2]+(x-old[:2])*(b[2:]-b[:2])/(old[2:]-old[:2]) for b in boxes];assert np.all(q[0]!=q[1],axis=1).sum()==0;assert np.all(np.linalg.norm(q[0]-q[1],axis=1)>0)
    write('math.json',dict(passed=True,reused_affine_inverse_and_interpolation_math=str((OLD/'math.json').relative_to(ROOT)),different_candidates_read_different_points=True,shared_previous_cells_do_not_force_equal_current_query=True,maximum_example_query_difference=float(np.max(abs(q[0]-q[1]))),not_an_identity_proof=True))
def register():
    assert read(OUT/'math.json')['passed'];p=read(BASE/'protocol.json');s.verify(p);s.verify(read(BASE/'maps_complete.json'));files=[Path(__file__),OUT/'math.json',OLD/'math.json',OLD/'readiness.json',BASE/'protocol.json',BASE/'results.json',BASE/'summary.json',BASE/'maps_complete.json',ROOT/'tracking/observed_spatial_reader.py',ROOT/'tracking/hex_linear_interpolation.py',ROOT/'tracking/channel_balanced_correspondence.py',ROOT/'tracking/box_normalized_correspondence.py',ROOT/'tracking/own_template_verification.py']
    write('protocol.json',dict(registered=s.stamp(),sources={f.relative_to(ROOT).as_posix():sha(f) for f in files},cases=[c['uid'] for c in p['cases']],scope='Exploratory WHOLE-method replication on already seen12 cases. Old coordinate-isolation failed5/12 and stays failed; no score on its failed subset. This question permits different support from the old prediction method, and cannot attribute any gain solely to coordinates.',method='Use ALL reference STATE-box cells. Per candidate T_k(x)=Bk_lo+(x-B0_lo)*(Bk_size/B0_size). Same original19 physical-pixel offsets, no new scale clip/anchor/threshold. Intersect full19 valid/hull masks across the TWO RELATIVE candidate transforms ONLY. Both candidates and all7 representations use exactly the SAME previous cells, but different current query tensors. Minimum3 unchanged.',history='Box-relative coordinates already tested with older nearest/global C8/R1/image/DIS method and failed full-tracker benefit. This tests their combination with continuous exact-constant interpolation and frozen channel-balanced full34 reader. It is not a new idea or a revision of the previous failed isolation criterion.',controls='Reuse original reader and local native cost; all7 existing FlyVis/input representations with identical new support/query. Additional no-FlyVis centered-shape IoU compares reference/current dimensions after removing centers, no threshold. Geometry and resupport are part of this method, not credited to neural core.',readiness='At least8/12 queries have >=3 common reference cells before any new real score. All12 remain denominator, unsupported=abstention0; no selective deletion.',metrics='Fixed offset9 and maximum over19; ties0.5, abstention0. Primary C34 maximum correctness minus original local cost and both34-channel input controls, paired5000 video bootstrap seed42. Exploratory gate requires positive95%lower against all3 and zero unsupported. A pass still needs distinct frozen cases/actual global ID fate before any deployment.',acceptance=[dict(criterion='Math and identical common previous cells across relative candidates/representations, sources/maps stable, no new model or threshold.',met=None),dict(criterion='At least8 supported queries before scoring.',met=None),dict(criterion='Exploratory C34 lower95% paired correctness difference positive versus original cost and both34-input controls, without unsupported queries.',met=None)],stop='If readiness fails no scoring. If gate fails stop, do not choose favorable channel subset or change shift/support thresholds. No full tracker or ID intervention.'))
    path=ROOT/'experiments/project_history.json';cat=read(path);assert not any(c['id']==OUT.name for c in cat['experiments']);cat['experiments'].append(dict(id=OUT.name,title='Один прошлый шаблон для двух рамок — относительные координаты и совместный считыватель',area='flyvis',state='running',started='2026-11-09',question='Поможет ли вся комбинация относительного шаблона с нынешним считывателем при одинаковых прошлых ячейках обеих рамок?',hypothesis='Зависимые от кандидата текущие координаты устраняют вынужденное равенство запросов; ошибку личности это само по себе не исключает.',expected='Ограниченная исследовательская проверка на готовых картах, с контролем изображения и геометрии. Предыдущий неудачный допуск сравнения одного выравнивания не пересматривается.',acceptance=read(OUT/'protocol.json')['acceptance'],evidence=[dict(kind='file',ref=f'runs/{OUT.name}/protocol.json',claim='Математика и критерии до новых оценок')]))
    path.write_text(json.dumps(cat,ensure_ascii=False,indent=2)+'\n',encoding='utf8');s.journal('Зафиксирована отдельная проверка полного относительного шаблона',f'*Постановка (`{OUT.name}`).* Математически проверено: одинаковые прошлые ячейки при разных относительных преобразованиях читают разные текущие точки. Это проверка всей комбинации относительного выравнивания, нового участка и зафиксированного считывателя, а не повтор остановленного сравнения одной геометрической поправки. Его допуск 5/12 остаётся не пройден. Прежний относительный метод уже проверялся с другим считывателем, новая идея ему не приписывается. Те же 12 просмотренных случаев, семь представлений, 19 сдвигов, новые модели не запускаются. До оценок требуется 8 поддержанных запросов. Контроли: исходная стоимость, одинаковые признаки входа глаза и чистая согласованность размеров. Источник `runs/{OUT.name}/protocol.json`.')
def run():
    p=read(OUT/'protocol.json');s.verify(p);bp=read(BASE/'protocol.json');s.verify(bp);s.verify(read(BASE/'maps_complete.json'));g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');reader=ObservedSpatialReader(g['points'],g['eye_size'],[1280,720]);nb=HexGeometry(np.array(g['points']),g['eye_size']).neighbors;plans=[];ready=0
    for c in bp['cases']:
        rc=np.array(c['sequence'][c['reference_relative']][1]);ct=np.array(c['sequence'][-1][1]);old=np.array(c['reference_raw']['state_box']);x=reader.points*(rc[2:]-rc[:2])/g['eye_size']+rc[:2];inside=((x>=old[:2])&(x<=old[2:])).all(1);queries=[];keep=inside.copy();shapes=[]
        for k in c['candidates']:
            box=np.array(k['box']);off=np.array(k['plan']['offsets']);trial=box[:2]+(x-old[:2])*(box[2:]-box[:2])/(old[2:]-old[:2]);trial=trial[:,None]+off;query=(trial-ct[:2])/((ct[2:]-ct[:2])/g['eye_size']);keep&=((trial>=box[:2])&(trial<=box[2:])).all((1,2))&(reader.mesh.find_simplex(query)>=0).all(1);queries.append(query)
            a0=old[2:]-old[:2];a1=box[2:]-box[:2];inter=np.minimum(a0,a1).prod();shapes.append(float(inter/(a0.prod()+a1.prod()-inter)))
        idx=np.flatnonzero(keep);supported=len(idx)>=3;ready+=supported;plans.append(dict(uid=c['uid'],video=c['video'],cells=idx,supported=supported,plans=[dict(cells=idx,query=q[idx]) for q in queries],shape_iou=shapes))
    write('readiness.json',dict(passed=ready>=8,supported_queries=ready,total=12,plans=plans));print('READY',ready,12,flush=True)
    if ready<8:return
    def features(key):
        a=s.arrays(key,bp);c=a['C34'].astype(float);r=a['R1'].astype(float);cen=r[:,s.SELECT[:4]]
        return dict(C34=c,FlyVis=c[:,4:12],R134_temporal=r,R134_spatial=neighbor_aug(r[:,:17].T[None],nb)[0].T,input8_temporal=r[:,s.SELECT],center8=np.concatenate([cen,cen],1),neighbor8=neighbor_aug(cen.T[None],nb)[0].T)
    records=[]
    for c,plan in zip(bp['cases'],plans):
        ref,cur=[features(k) for k in c['keys']];variants={}
        for name in s.NAMES:
            vals=[]
            for k,subplan in zip(c['candidates'],plan['plans']):
                q=reader.compare(ref[name],cur[name],subplan);vals.append(dict(kind=k['kind'],fixed=q['scores'][9] if q['scores'] is not None else None,maximum=q['score'],winner=q['winner'],scores=q['scores']))
            variants[name]=vals
        records.append(dict(uid=c['uid'],video=c['video'],supported=plan['supported'],cells=len(plan['cells']),variants=variants,shape=[dict(kind=k['kind'],score=sh) for k,sh in zip(c['candidates'],plan['shape_iou'])]))
    s.verify(p);write('results.json',dict(passed=True,records=records,no_new_maps=True,no_threshold_tuning=True));print('Relative joint reader complete',flush=True)
if __name__=='__main__':{'math':math,'register':register,'run':run}[sys.argv[1]]()
