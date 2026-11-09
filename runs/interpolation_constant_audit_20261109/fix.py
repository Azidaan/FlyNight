"""Freeze/test a constant-preserving fix and compare only new numerical scores."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import ast,hashlib,json,sys
from pathlib import Path
from datetime import datetime,timedelta,timezone
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;BASE=ROOT/'runs/common_state_memory_20261109';MSK=timezone(timedelta(hours=3));STOP=datetime(2026,11,9,6,35,tzinfo=MSK)
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
remaining=(STOP-datetime.now(MSK)).total_seconds()
if remaining<=0:raise SystemExit('Save/stop deadline')
guard_entrypoint(timeout=min(1200,remaining))
import numpy as np
from tracking.hex_linear_interpolation import sample_linear
from tracking.observed_spatial_reader import ObservedSpatialReader
from tracking.channel_balanced_correspondence import channel_balanced_similarity
from tracking.z13_readout import HexGeometry
from scripts.z13_target_model import neighbor_aug
OTHER=[0,1,2,3]+list(range(12,34));SELECT=[0,5,11,16,17,22,28,33]
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def cast(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,dict):return {k:cast(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [cast(x) for x in v]
    return v
def write(n,v):
    with (OUT/n).open('x',encoding='utf8') as f:json.dump(cast(v),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def verify(p):
    for f,h in p['sources'].items():assert sha(ROOT/f)==h,f
def verify_before(p):
    substitutions={}
    for f,h in p['sources'].items():
        path=ROOT/f
        if f in ['tracking/observed_spatial_reader.py','tracking/hex_background_filter.py']:
            path=OUT/'before'/Path(f).name;substitutions[f]=path.relative_to(ROOT).as_posix()
        assert sha(path)==h,f
    return substitutions
def register():
    probe=read(OUT/'protocol.json');sub=verify_before(probe);assert not read(OUT/'results.json')['invariant_met']
    sources=[Path(__file__),OUT/'protocol.json',OUT/'results.json',OUT/'before/observed_spatial_reader.py',OUT/'before/hex_background_filter.py',ROOT/'tracking/observed_spatial_reader.py',ROOT/'tracking/hex_background_filter.py',ROOT/'tracking/hex_linear_interpolation.py',ROOT/'tests/test_hex_linear_interpolation.py',ROOT/'tests/test_observed_spatial_reader.py',ROOT/'tests/test_hex_background_filter.py',BASE/'protocol.json',BASE/'results.json',BASE/'summary.json',BASE/'maps_complete.json',BASE/'recover_maps.py',BASE/'compare.py']
    write('fix_protocol.json',dict(registered=datetime.now(MSK).isoformat(),before_sources_verified_via_archives=sub,method='Triangle interpolant v2+b0*(v0-v2)+b1*(v1-v2), same barycentric coordinates and hull. Constant channels exactly preserved. Global unit_patch sets exactly constant channels to zero before normalization, as balanced reader already did. No variance cutoff, epsilon, threshold, weights, geometry or search change.',math='Exact partition of unity: sum(b_i)=1 gives the same affine interpolant. Difference expression multiplies zeros for a constant field, avoiding roundoff-generated contrast. Constant-current NCC must abstain; affine fields and source-node values must agree to normal floating error. This corrects implementation consistency, not proof of an identity-bearing signal.',checks='Run3new invariants plus directly affected reader/background-filter tests. Reuse unchanged channel-balancing proof by source hash, no GPU/model/detector. Repeat the NEW constant scenario with fixed expression; prior old76grids read, not recalculated.',real_regression='Same19 frozen real geometries, maps, templates, controls and six arms as common_state_memory. Only numerical interpolation/constant-centering changes. Read all ready old798grids; evaluate fixed grids without neural replay. Report maximum score change, null transitions, AUROC/ranking changes and whole support. Development/regression evidence only, not an independent new identity benchmark.',sources={p.relative_to(ROOT).as_posix():sha(p) for p in sources}))
    path=ROOT/'experiments/project_history.json';cat=read(path);card=next(c for c in cat['experiments'] if c['id']==OUT.name);card['acceptance'][0].update(met=False,evidence=f'runs/{OUT.name}/results.json');card['acceptance']+=[dict(criterion='Дописано до исправленной проверки: постоянный сигнал сохраняется точно, ложных оценок0; аффинный перенос и затронутые тесты проходят. Причина —1440ложныхчиселвновомстрогомпримере.',met=None),dict(criterion='Дописано до сравнения: реальные карты/геометрия/веса неизменны, новый численный результат сопоставлен со всеми готовыми798сетками без нейронного повтора; величина эффекта не объявляетсяIDвыигрышем.',met=None)];card['evidence'].append(dict(kind='file',ref=f'runs/{OUT.name}/fix_protocol.json',claim='Формула и численная сверка до новых исправленных результатов'));path.write_text(json.dumps(cat,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print('Registered exact-constant fix',flush=True)
def tests():
    p=read(OUT/'fix_protocol.json');verify(p)
    import pytest
    assert pytest.main(['tests/test_hex_linear_interpolation.py','tests/test_observed_spatial_reader.py','tests/test_hex_background_filter.py','-q'])==0
    g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');reader=ObservedSpatialReader(g['points'],g['eye_size'],[1280,720]);x,y=reader.points.T;ref=np.column_stack([np.sin(x/(17+c))+np.cos(y/(23+c))+np.sin(x*y/(301+13*c)) for c in range(8)]);old=read(OUT/'results.json');by={(c['video'],c['frame'],c['old']):c for c in read(BASE/'protocol.json')['cases']};records=[]
    for r in old['records']:
        c=by[(r['video'],r['frame'],r['old'])];plan=c['commonplan'];constants=r['level']*(1+np.arange(8)/10);current=np.tile(constants,(721,1));sampled=sample_linear(reader.mesh,current,np.array(plan['query']));assert np.array_equal(sampled,np.broadcast_to(constants,sampled.shape));result=reader.compare(ref,current,plan);assert all(v is None for v in result['scores']);records.append(dict(video=r['video'],frame=r['frame'],old=r['old'],level=r['level'],scores=result['scores'],maximum_value_error=0.))
    verify(p);write('fixed_invariant.json',dict(passed=True,old_nonnull=old['nonnull_scores'],new_nonnull=0,grids=len(records),records=records,constant_preserved_exactly=True));print('Fixed constant invariant passed',flush=True)
def features(k,p,nb):
    # Load only existing saved arrays. No import of old executable study modules.
    ns=dict(ROOT=ROOT,OUT=BASE,np=np,read=read,sha=sha,OTHER=OTHER);path=BASE/'recover_maps.py';nodes=[n for n in ast.parse(path.read_text(encoding='utf8')).body if isinstance(n,ast.FunctionDef) and n.name in ['ready_array','load']];exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns);a=ns['load'](k,p);c=a['C34'].astype(float);r=a['R1'].astype(float);central=r[:,SELECT[:4]]
    return dict(C34=c,FlyVis=c[:,4:12],R134_temporal=r,R134_spatial=neighbor_aug(r[:,:17].T[None],nb)[0].T,input8_temporal=r[:,SELECT],center8=np.concatenate([central,central],1),neighbor8=neighbor_aug(central.T[None],nb)[0].T)
def regression():
    fp=read(OUT/'fix_protocol.json');verify(fp);assert read(OUT/'fixed_invariant.json')['passed'];p=read(BASE/'protocol.json');sub=verify_before(p);verify(read(BASE/'maps_complete.json'));old=read(BASE/'results.json');g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');reader=ObservedSpatialReader(g['points'],g['eye_size'],[1280,720]);nb=HexGeometry(np.array(g['points']),g['eye_size']).neighbors;records=[];delta=0.;transitions=0;newgrids=0;aliased=0
    for c in p['cases']:
        original=next(r for r in old['records'] if all(r[k]==c[k] for k in ['video','frame','old','new']));r={k:original[k] for k in ['video','frame','old','new','kind','supported','clipped','early_equals_last']};r['variants']={}
        if not r['supported']:records.append(r);continue
        maps={name:features(k,p,nb) for name,k in c['snapshots'].items()};cells=c['commonplan']['cells'];eq=np.array(c['early_query']);bq=np.array(c['roundtrip_b_query']);valid=np.array(c['roundtrip_valid_nodes']);q=np.array(c['commonplan']['query'])
        for name in maps['B_last']:
            al=maps['A_last'][name];ac=maps['A_current'][name];bl=maps['B_last'][name];bc=maps['B_current'][name];be=maps['B_early'][name];values={}
            def pack(scores):
                ids=[i for i,v in enumerate(scores) if v is not None];winner=max(ids,key=lambda i:scores[i]) if ids else None
                return dict(scores=scores,fixed=scores[9],maximum=scores[winner] if winner is not None else None,winner=winner)
            def grid(a,b,plan):return pack(reader.compare(a,b,plan)['scores'])
            values['A_native']=grid(al,ac,c['bplan']);newgrids+=1
            if c['snapshots']['A_last']==c['snapshots']['B_last'] and c['snapshots']['A_current']==c['snapshots']['B_current']:values['B_native']=values['A_native'];aliased+=1
            else:values['B_native']=grid(bl,bc,c['bplan']);newgrids+=1
            if np.array_equal(cells,c['bplan']['cells']):values['A_common']=values['A_native'];values['B_common']=values['B_native'];aliased+=2
            else:values['A_common']=grid(al,ac,c['commonplan']);values['B_common']=grid(bl,bc,c['commonplan']);newgrids+=2
            current=sample_linear(reader.mesh,bc,q).transpose(1,0,2);early=sample_linear(reader.mesh,be,eq);projected=np.zeros_like(bl);projected[valid]=sample_linear(reader.mesh,bl,bq[valid]);back=sample_linear(reader.mesh,projected,eq)
            values['C_common']=pack([channel_balanced_similarity(early,v) for v in current]);values['B_roundtrip']=pack([channel_balanced_similarity(back,v) for v in current]);newgrids+=2
            for arm,value in values.items():
                before=original['variants'][name]['values'][arm]['scores']
                for a,b in zip(before,value['scores']):
                    if (a is None)!=(b is None):transitions+=1
                    elif a is not None:delta=max(delta,abs(a-b))
            r['variants'][name]=values
        records.append(r)
    table={}
    for metric in ['fixed','maximum']:
        table[metric]={}
        for name in ['C34','FlyVis','R134_temporal','R134_spatial','input8_temporal','center8','neighbor8']:
            table[metric][name]={}
            for arm in ['A_native','B_native','A_common','B_common','B_roundtrip','C_common']:
                a=[r['variants'][name][arm][metric] for r in records if r['supported'] and r['kind']=='own'];b=[r['variants'][name][arm][metric] for r in records if r['supported'] and r['kind']=='foreign'];a=[v for v in a if v is not None];b=[v for v in b if v is not None];num=sum(float(x>y)+.5*float(x==y) for x in a for y in b);den=len(a)*len(b);table[metric][name][arm]=dict(auc=num/den if den else None,correct_rankings=num,total_rankings=den)
    verify(fp);verify_before(p);write('real_regression.json',dict(passed=True,records=records,table=table,maximum_score_difference=delta,null_transitions=transitions,new_numeric_grids=newgrids,identical_grid_aliases=aliased,ready_old_grids_compared=798,old_sources_verified_via_archives=sub,zero_neural_retinal_detector_tracker_calls=True));print('Real numerical regression',delta,transitions,table['maximum']['C34'],flush=True)
if __name__=='__main__':{'register':register,'tests':tests,'regression':regression}[sys.argv[1]]()
