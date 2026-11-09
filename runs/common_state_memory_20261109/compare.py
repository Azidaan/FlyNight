"""Fixed common-grid comparison, with interpolation-only and no-FlyVis controls."""
import recover_maps as m
from study import *
from tracking.channel_balanced_correspondence import channel_balanced_similarity
from tracking.z13_readout import HexGeometry
from scripts.z13_target_model import neighbor_aug
NAMES=['C34','FlyVis','R134_temporal','R134_spatial','input8_temporal','center8','neighbor8']
ARMS=['A_native','B_native','A_common','B_common','B_roundtrip','C_common']
def eventkey(c):return c['video'],c['frame'],c['old'],c['new']
def register():
    p=read(OUT/'protocol.json');verify(p);verify(read(OUT/'maps_complete.json'));files=[Path(__file__),OUT/'protocol.json',OUT/'map_protocol.json',OUT/'maps_complete.json',OUT/'mathematical_checks.json',OUT/'recover_maps.py']
    ready={};oldp=read(ROOT/'runs/frozen_reader_transfer_20261109/protocol.json');oldr=read(ROOT/'runs/frozen_reader_transfer_20261109/results.json')
    for c in p['cases']:
        old=next((x for x in oldp['cases'] if eventkey(c)==eventkey(x)),None)
        if old is None:continue
        if old['plan_joint']['cells']!=c['bplan']['cells'] or not np.array_equal(old['plan_joint']['query'],c['bplan']['query']):continue
        assert old['map_keys']==[c['snapshots']['A_last'],c['snapshots']['A_current']]
        rr=next(x for x in oldr['new_records'] if eventkey(x)==eventkey(c));ready['|'.join(map(str,eventkey(c)))]=rr['variants']
    files.extend([ROOT/'runs/frozen_reader_transfer_20261109/protocol.json',ROOT/'runs/frozen_reader_transfer_20261109/results.json',ROOT/'runs/direct_memory_anchor_inventory_20261109/results.json',ROOT/'runs/direct_memory_anchor_inventory_20261109/protocol.json',ROOT/'runs/bbox_confirmation_inventory_20261108/protocol.json',ROOT/'runs/clipped_anchor_readout_20261109/study.py'])
    source=read(ROOT/'runs/direct_memory_anchor_inventory_20261109/results.json');parent=read(ROOT/'runs/direct_memory_anchor_inventory_20261109/protocol.json');context={}
    ns=dict(np=np);path=ROOT/'runs/clipped_anchor_readout_20261109/study.py';node=next(n for n in ast.parse(path.read_text(encoding='utf8')).body if isinstance(n,ast.FunctionDef) and n.name=='raw_observation');exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
    cfg=read(ROOT/'runs/bbox_confirmation_inventory_20261108/protocol.json')['native_config']
    for c in p['cases']:
        r=next(x for x in source['records'] if eventkey(x)==eventkey(c));info=parent['videos'][c['video']]
        with np.load(ROOT/info['native'],allow_pickle=False) as z:native=z['outputs']
        with np.load(ROOT/info['raw'],allow_pickle=False) as z:raw={n:z[n] for n in ['frame_idx','conf','cls','xyxy']}
        last=ns['raw_observation'](native,raw,c['old'],c['last_frame'],cfg);b=last['box'];lastclip=b[0]<=0 or b[1]<=0 or b[2]>=1280 or b[3]>=720
        context['|'.join(map(str,eventkey(c)))]=dict(geometry=dict(forward_iou=r['candidate']['forward_iou'],negative_native_cost=-r['candidate']['cost'],common_cell_count=len(c['commonplan']['cells'])),last_raw_box=b,last_clipped=bool(lastclip),any_snapshot_clipped=c['clipped'] or bool(lastclip))
    write('comparison_protocol.json',dict(registered=stamp(),ready_native=ready,context=context,method='Frozen channel-normalized19-shift grids, shared common sampling and same current tensor for B/C. Original A_native exact ready grids reused. B_native/other identical map-plan aliases reused instead of recalculated. All NEW scores independently verified with SciPy interpolation and a separate scalar Pearson expression<=1e-9.',metrics='AUROC before offset9/after maximum; same supported defined events for each comparison. Paired5000 video bootstrap seed42 and valid resamples, no decision threshold. Primary early-vs-latest is C_common/B_common plus interpolation control B_roundtrip; native/common difference is separately reported. Supplementary geometry-only rankings are unchanged native forwardIoU, negative native cost, and common cell count; no fitted combination or threshold. Clip strata include early/latest/current raw observations, the parent marked early/current only.',sources={p.relative_to(ROOT).as_posix():sha(p) for p in files}))
def features(k,p,nb):
    a=m.load(k,p);c=a['C34'].astype(float);r=a['R1'].astype(float);central=r[:,SELECT[:4]]
    return dict(C34=c,FlyVis=c[:,4:12],R134_temporal=r,R134_spatial=neighbor_aug(r[:,:17].T[None],nb)[0].T,input8_temporal=r[:,SELECT],center8=np.concatenate([central,central],1),neighbor8=neighbor_aug(central.T[None],nb)[0].T)
def pack(scores):
    if scores is None:return dict(scores=None,fixed=None,maximum=None,winner=None)
    ids=[i for i,v in enumerate(scores) if v is not None];winner=max(ids,key=lambda i:scores[i]) if ids else None
    return dict(scores=scores,fixed=scores[9],maximum=scores[winner] if winner is not None else None,winner=winner)
def independent(a,b):
    x=a-a.mean(0);y=b-b.mean(0);x[:,np.ptp(a,axis=0)==0]=0;y[:,np.ptp(b,axis=0)==0]=0
    xn=np.sqrt(np.sum(x*x,axis=0));yn=np.sqrt(np.sum(y*y,axis=0));active=np.flatnonzero(xn>0)
    if not len(active) or not np.any(yn):return None
    return sum(float(np.sum(x[:,j]*y[:,j])/(xn[j]*yn[j])) if yn[j]>0 else 0. for j in active)/len(active)
def run():
    cp=read(OUT/'comparison_protocol.json');verify(cp);p=read(OUT/'protocol.json');verify(p);verify(read(OUT/'maps_complete.json'))
    if (OUT/'results.json').exists():raise SystemExit('Ready results exist; never repeat')
    g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');reader=ObservedSpatialReader(g['points'],g['eye_size'],[1280,720]);nb=HexGeometry(np.array(g['points']),g['eye_size']).neighbors;records=[];maximum=0.;computed=0;reused=0
    for c in p['cases']:
        cells=c['commonplan']['cells'];r={k:c[k] for k in ['video','frame','old','new','kind','clipped','reference_frame','reference_age','last_frame','early_equals_last']};r.update(common_cells=cells,native_cells=c['bplan']['cells'],variants={},supported=len(cells)>=3);r['context']=cp['context']['|'.join(map(str,eventkey(c)))];r['clipped']=r['context']['any_snapshot_clipped']
        if len(cells)<3:records.append(r);continue
        snapshots={k:features(v,p,nb) for k,v in c['snapshots'].items()};eq=np.array(c['early_query']);bq=np.array(c['roundtrip_b_query']);valid=np.array(c['roundtrip_valid_nodes']);si=reader.mesh.find_simplex(eq);assert valid[reader.mesh.simplices[si]].all()
        query=np.array(c['commonplan']['query']);native=np.array(c['bplan']['query']);old=cp['ready_native'].get('|'.join(map(str,eventkey(c))));cache={}
        for name in NAMES:
            values={};prov={}
            def grid(a,b,plan,tag):
                nonlocal computed,maximum
                template=a[plan['cells']];patch=LinearNDInterpolator(reader.mesh,b)(np.array(plan['query'])).transpose(1,0,2);return evidence(template,patch,tag)
            def evidence(template,patch,tag):
                nonlocal computed,maximum
                scores=[channel_balanced_similarity(template,v) for v in patch];ind=[independent(template,v) for v in patch]
                for a,b in zip(scores,ind):
                    assert (a is None)==(b is None)
                    if a is not None:maximum=max(maximum,abs(a-b));assert abs(a-b)<=1e-9
                computed+=1;prov[tag]='new_missing_grid';return pack(scores)
            al=snapshots['A_last'][name];ac=snapshots['A_current'][name];bl=snapshots['B_last'][name];bc=snapshots['B_current'][name];be=snapshots['B_early'][name]
            if old:
                values['A_native']=pack(old[name]['joint']['scores']);prov['A_native']='runs/frozen_reader_transfer_20261109/results.json';reused+=1
            else:values['A_native']=grid(al,ac,c['bplan'],'A_native')
            if c['snapshots']['A_last']==c['snapshots']['B_last'] and c['snapshots']['A_current']==c['snapshots']['B_current']:
                values['B_native']=values['A_native'];prov['B_native']='identical maps/plan A_native';reused+=1
            else:values['B_native']=grid(bl,bc,c['bplan'],'B_native')
            same=np.array_equal(cells,c['bplan']['cells'])
            if same:values['A_common']=values['A_native'];values['B_common']=values['B_native'];prov['A_common']='identical A_native support';prov['B_common']='identical B_native support';reused+=2
            else:values['A_common']=grid(al,ac,c['commonplan'],'A_common');values['B_common']=grid(bl,bc,c['commonplan'],'B_common')
            # One current tensor object, reused verbatim for the two templates.
            current=LinearNDInterpolator(reader.mesh,bc)(query).transpose(1,0,2)
            sampled_early=LinearNDInterpolator(reader.mesh,be)(eq)
            projected=np.zeros_like(bl);projected[valid]=LinearNDInterpolator(reader.mesh,bl)(bq[valid]);back=LinearNDInterpolator(reader.mesh,projected)(eq)
            assert np.isfinite(back).all() and np.isfinite(sampled_early).all()
            exact_same_reference=c['snapshots']['B_early']==c['snapshots']['B_last'] and np.array_equal(eq,reader.points[cells]) and np.array_equal(bq[valid],reader.points[valid])
            if np.array_equal(cells,c['bplan']['cells']) and exact_same_reference:
                assert c['snapshots']['B_early']==c['snapshots']['B_last'];assert np.max(abs(sampled_early-bl[cells]))<1e-9;assert np.max(abs(back-bl[cells]))<1e-9
                values['C_common']=values['B_common'];values['B_roundtrip']=values['B_common'];prov['C_common']=prov['B_roundtrip']='identical reference timestamp and sampling';reused+=2
            else:values['C_common']=evidence(sampled_early,current,'C_common');values['B_roundtrip']=evidence(back,current,'B_roundtrip')
            r['variants'][name]=dict(values=values,provenance=prov,reference_active_channels=dict(B=int(np.sum(np.ptp(bl[cells],axis=0)>0)),C=int(np.sum(np.ptp(sampled_early,axis=0)>0))),current_samples_B_C_identical=True)
        records.append(r);status('comparison',events_done=len(records),total=len(p['cases']),new_grids=computed,reused_grids=reused);print('SCORED',eventkey(c),len(cells),flush=True)
    verify(cp);verify(p);write('results.json',dict(passed=True,records=records,new_grids=computed,reused_grids=reused,independent_new_score_max_error=maximum,zero_threshold_tuning=True,zero_id_assignment=True,all_current_samples_B_C_identical=True));status('comparison_complete',events=len(records))
def auc(rows,name,arm,metric,weights=None):
    good=[(r,r['variants'][name]['values'][arm][metric]) for r in rows if r['supported']];good=[(r,v) for r,v in good if v is not None];own=[(r,v) for r,v in good if r['kind']=='own'];other=[(r,v) for r,v in good if r['kind']=='foreign'];num=0.;den=0.
    for a,x in own:
        for b,y in other:
            w=1 if weights is None else weights.get(a['video'],0)*weights.get(b['video'],0);num+=w*(float(x>y)+.5*float(x==y));den+=w
    return num/den if den else None,num,den
def analyze():
    cp=read(OUT/'comparison_protocol.json');verify(cp);d=read(OUT/'results.json');rows=d['records'];videos=sorted({r['video'] for r in rows if r['supported']});rng=np.random.default_rng(42);draws=rng.integers(0,len(videos),(5000,len(videos)));counts=np.array([np.bincount(v,minlength=len(videos)) for v in draws]);weights=[dict(zip(videos,map(int,v))) for v in counts];table={};differences={}
    def common_defined(rr,names,arms,metric):return [r for r in rr if r['supported'] and all(r['variants'][n]['values'][a][metric] is not None for n in names for a in arms)]
    for scope,rr in [('all',rows),('earlier_only',[r for r in rows if not r['early_equals_last']]),('clipped',[r for r in rows if r['clipped']]),('unclipped',[r for r in rows if not r['clipped']])]:
        table[scope]={}
        for metric in ['fixed','maximum']:
            valid=common_defined(rr,NAMES,ARMS,metric);table[scope][metric]=dict(events=len(valid),own=sum(r['kind']=='own' for r in valid),foreign=sum(r['kind']=='foreign' for r in valid),values={})
            for name in NAMES:
                table[scope][metric]['values'][name]={arm:dict(zip(['auc','correct_rankings','total_rankings'],auc(valid,name,arm,metric))) for arm in ARMS}
    for name in NAMES:
        for first,second in [('C_common','B_common'),('C_common','B_roundtrip'),('B_native','A_native')]:
            valid=common_defined(rows,[name],[first,second],'maximum');a=auc(valid,name,first,'maximum')[0];b=auc(valid,name,second,'maximum')[0];vals=[]
            for w in weights:
                aa=auc(valid,name,first,'maximum',w)[0];bb=auc(valid,name,second,'maximum',w)[0]
                if aa is not None and bb is not None:vals.append(aa-bb)
            differences[f'{name}:{first}-{second}']=dict(point=a-b if a is not None and b is not None else None,ci95=np.percentile(vals,[2.5,97.5]).tolist() if vals else None,valid_resamples=len(vals),total_resamples=5000)
    for control in ['R134_temporal','R134_spatial']:
        valid=common_defined(rows,['C34',control],['C_common'],'maximum');a=auc(valid,'C34','C_common','maximum')[0];b=auc(valid,control,'C_common','maximum')[0];vals=[]
        for w in weights:
            aa=auc(valid,'C34','C_common','maximum',w)[0];bb=auc(valid,control,'C_common','maximum',w)[0]
            if aa is not None and bb is not None:vals.append(aa-bb)
        differences['C34:'+control]=dict(point=a-b if a is not None and b is not None else None,ci95=np.percentile(vals,[2.5,97.5]).tolist() if vals else None,valid_resamples=len(vals),total_resamples=5000)
    geom={};valid=common_defined(rows,NAMES,ARMS,'maximum')
    for name in ['forward_iou','negative_native_cost','common_cell_count']:
        own=[r['context']['geometry'][name] for r in valid if r['kind']=='own'];other=[r['context']['geometry'][name] for r in valid if r['kind']=='foreign'];num=sum(float(a>b)+.5*float(a==b) for a in own for b in other);den=len(own)*len(other);geom[name]=dict(auc=num/den if den else None,correct_rankings=num,total_rankings=den)
    write('summary.json',dict(passed=True,table=table,differences=differences,geometry_only=geom,selection='Unchanged22 direct-earlier/current pairs; unsupported retained; last-reference identity not generally directly known.',new_grids=d['new_grids'],reused_grids=d['reused_grids'],independent_error=d['independent_new_score_max_error'],no_tracker_or_id_fate_claim=True));print('SUMMARY',json.dumps(table['all'],ensure_ascii=False),flush=True)
if __name__=='__main__':{'register':register,'run':run,'analyze':analyze}[sys.argv[1]]()
