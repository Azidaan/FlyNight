"""Frozen descriptive transfer to the three remaining unscored videos."""
import sys,json,ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;BASE=ROOT/'runs/historical_two_candidate_comparison_20261109';REL=ROOT/'runs/relative_joint_reader_20261109';DATA=ROOT/'runs/historical_direct_query_inventory_20261109'
sys.path.insert(0,str(BASE));import study_v2 as s
import numpy as np
from tracking.observed_spatial_reader import ObservedSpatialReader
from tracking.z13_readout import HexGeometry
from scripts.z13_target_model import neighbor_aug
read=s.read;sha=s.sha
def write(n,v):
    with (OUT/n).open('x',encoding='utf8') as f:json.dump(s.cast(v),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def verify(p):
    for name,h in p['sources'].items():assert sha(ROOT/name)==h,name
def namespace():
    # Reuse the exact already audited restoration and map-loading functions.
    # Their own globals deliberately point at THIS protocol/output directory.
    ns=dict(s.__dict__);ns.update(OUT=OUT,DATA=DATA,write=write,verify=verify)
    source=BASE/'study_v2.py';tree=ast.parse(source.read_text(encoding='utf8'));nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['arrays','restore']];assert len(nodes)==2;exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns);return ns
def register():
    old=read(BASE/'protocol.json');s.verify(old);assert read(REL/'audit.json')['passed'];inv=read(DATA/'results.json');dp=read(DATA/'protocol.json');s.verify(dp);seen={c['video'] for c in old['cases']};cases=[]
    for video in dp['videos']:
        good=[c for c in inv['records'] if c['clean'] and c['video']==video]
        if good and video not in seen:cases.append(min(good,key=lambda c:c['uid']))
    assert len(cases)==3 and len({c['video'] for c in cases})==3
    requests={};sources=dict(old['sources'])
    for case in cases:
        case['keys']=[]
        for seq in [case['sequence'][:case['reference_relative']+1],case['sequence']]:
            key=s.seqkey(case['video'],seq);case['keys'].append(key);req=dict(video=case['video'],sequence=seq,ready={})
            paths=sorted((ROOT/'runs').glob(f'*/maps/{key}.npz'))+sorted((ROOT/'runs').glob(f'*/features/{key}.npz'))+sorted((ROOT/'runs').glob(f'*/{key}.npz'))
            for path in paths:
                with np.load(path,allow_pickle=False) as z:
                    for name in ['C34','C','C_OTHER','R1']:
                        if name not in z.files:continue
                        a=z[name]
                        if a.shape not in [(721,8),(721,26),(721,34)]:continue
                        slot='C8' if name=='C' and a.shape[1]==8 else ('C34' if name=='C' else name)
                        if slot not in req['ready']:req['ready'][slot]=dict(path=path.relative_to(ROOT).as_posix(),array=name);sources[path.relative_to(ROOT).as_posix()]=sha(path)
            requests[key]=req
        for name in [dp['videos'][case['video']]['data'],dp['videos'][case['video']]['raw'],dp['videos'][case['video']]['native'],dp['videos'][case['video']]['annotation'],dp['videos'][case['video']]['alignment']]:sources[name]=sha(ROOT/name)
    for path in [Path(__file__),BASE/'study_v2.py',BASE/'protocol.json',DATA/'protocol.json',DATA/'results.json',REL/'protocol.json',REL/'summary.json',REL/'audit.json']:sources[path.relative_to(ROOT).as_posix()]=sha(path)
    write('protocol.json',dict(registered=s.stamp(),cases=cases,requests=requests,sources=sources,selection='All three remaining clean videos from the fixed84-video inventory, first clean UID per video, none of previous12. No score consulted. Historical training videos used by older learned methods, not new official split.',question='Does the fixed whole relative-template method transfer to additional directly confirmed own/foreign queries? Primary C34 versus its old prediction-query reader, native local cost and matched inputs. No switch to favorable C8 primary.',method='Exactly relative_joint_reader_20261109: whole reference-state box, candidate-dependent affine transform, common previous cells across BOTH relative candidates, same19 physical offsets and unchanged3-cell minimum. Same seven feature controls; old original plans are scored too. No new scale/anchor/threshold/channel choice.',restore='Reuse already audited arrays/restore functions from bound study_v2.py, with explicit current namespace OUT/DATA/write/verify. Same archived fr_hex,2steps/dt1/60/fade1s; only missing6snapshot maps restored, any ready arrays retained. Model weights unchanged. Log restoration denominator12 belongs to reused function; true protocol cases/snapshots=3/6.',ready='All three must have >=3 common cells in relative method before restoring models. Otherwise stop without loosening support.',metrics='Descriptive correct/wrong/tie/abstain for all3, before/after19 shifts, old/relative and original cost. Report every query and matched image outcome. With only3 videos, no generalization/ID claim and no deployment regardless of result; no bootstrap interval sold as certainty.',acceptance=[dict(criterion='Three new videos selected without scores, direct9 raw-GT matches independently checked, common support/maps/source/weights verified.',met=None),dict(criterion='Old and relative C34 plus all matched inputs recorded on all3 without threshold tuning or selective exclusion.',met=None)],stop='Only this fixed descriptive transfer; do not retune on3 or add another reader. No tracker replay, ID action or full HOTA. Save/stop06:35; all own processes ended06:40.'))
    path=ROOT/'experiments/project_history.json';cat=read(path);assert not any(c['id']==OUT.name for c in cat['experiments']);cat['experiments'].append(dict(id=OUT.name,title='Относительный пространственный шаблон — перенос на три оставшихся видео',area='flyvis',state='running',started='2026-11-09',question='Сохраняется ли поведение зафиксированного относительного шаблона на трёх других подтверждённых запросах?',hypothesis='Общее выравнивание может помогать обоим видам признаков; отдельная добавка FlyVis не предполагается.',expected='Описательный перенос без смены основного C34, без новых порогов и повторения готовых карт. Три видео недостаточны для принятия нового трекера.',acceptance=read(OUT/'protocol.json')['acceptance'],evidence=[dict(kind='file',ref=f'runs/{OUT.name}/protocol.json',claim='Фиксация всех оставшихся новых видео до оценок')]))
    path.write_text(json.dumps(cat,ensure_ascii=False,indent=2)+'\n',encoding='utf8');s.journal('Зафиксирован перенос относительного шаблона на оставшиеся три видео',f'*Постановка (`{OUT.name}`).* Из прежнего фиксированного инвентаря взяты все три чистых видео, не входившие в просмотренные 12; первый запрос каждого. Метод и основной C34 не меняются. Старые и относительные координаты, все семь представлений и исходная стоимость. Новые пороги и модели обучения отсутствуют; восстанавливаются лишь недостающие снимки из готового входа глаза. Трёх видео недостаточно для принятия метода, результат описательный. Источник `runs/{OUT.name}/protocol.json`.')
def readiness():
    p=read(OUT/'protocol.json');verify(p);g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');reader=ObservedSpatialReader(g['points'],g['eye_size'],[1280,720]);plans=[]
    for c in p['cases']:
        rc=np.array(c['sequence'][c['reference_relative']][1]);ct=np.array(c['sequence'][-1][1]);old=np.array(c['reference_raw']['state_box']);x=reader.points*(rc[2:]-rc[:2])/g['eye_size']+rc[:2];keep=((x>=old[:2])&(x<=old[2:])).all(1);queries=[]
        for k in c['candidates']:
            b=np.array(k['box']);trial=b[:2]+(x-old[:2])*(b[2:]-b[:2])/(old[2:]-old[:2]);trial=trial[:,None]+np.array(k['plan']['offsets']);q=(trial-ct[:2])/((ct[2:]-ct[:2])/g['eye_size']);keep&=((trial>=b[:2])&(trial<=b[2:])).all((1,2))&(reader.mesh.find_simplex(q)>=0).all(1);queries.append(q)
        idx=np.flatnonzero(keep);plans.append(dict(uid=c['uid'],cells=idx,plans=[dict(cells=idx,query=q[idx]) for q in queries]))
    write('readiness.json',dict(passed=all(len(c['cells'])>=3 for c in plans),plans=plans));print('READY',[len(c['cells']) for c in plans],flush=True)
def restore():
    assert read(OUT/'readiness.json')['passed'];namespace()['restore']()
def compare():
    p=read(OUT/'protocol.json');verify(p);verify(read(OUT/'maps_complete.json'));plans=read(OUT/'readiness.json')['plans'];g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');reader=ObservedSpatialReader(g['points'],g['eye_size'],[1280,720]);nb=HexGeometry(np.array(g['points']),g['eye_size']).neighbors;load=namespace()['arrays'];records=[]
    def features(key):
        a=load(key,p);c=a['C34'].astype(float);r=a['R1'].astype(float);cen=r[:,s.SELECT[:4]]
        return dict(C34=c,FlyVis=c[:,4:12],R134_temporal=r,R134_spatial=neighbor_aug(r[:,:17].T[None],nb)[0].T,input8_temporal=r[:,s.SELECT],center8=np.concatenate([cen,cen],1),neighbor8=neighbor_aug(cen.T[None],nb)[0].T)
    for case,plan in zip(p['cases'],plans):
        ref,cur=[features(k) for k in case['keys']];variants={}
        for name in s.NAMES:
            vals=[]
            for candidate,relative in zip(case['candidates'],plan['plans']):
                modes={}
                for mode,geom in [('old',candidate['plan']),('relative',relative)]:
                    q=reader.compare(ref[name],cur[name],geom);modes[mode]=dict(fixed=q['scores'][9] if q['scores'] is not None else None,maximum=q['score'],scores=q['scores'],winner=q['winner'])
                vals.append(dict(kind=candidate['kind'],values=modes))
            variants[name]=vals
        records.append(dict(uid=case['uid'],video=case['video'],variants=variants,cost=[dict(kind=k['kind'],score=-k['original_cost']) for k in case['candidates']]))
    verify(p);write('results.json',dict(passed=True,records=records,no_threshold_tuning=True,no_ID_write=True));print('Three-video transfer complete',flush=True)
if __name__=='__main__':{'register':register,'readiness':readiness,'restore':restore,'compare':compare}[sys.argv[1]]()
