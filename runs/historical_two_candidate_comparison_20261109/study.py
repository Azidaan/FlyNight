"""Frozen within-query identity test on twelve archived direct-GT car choices."""
import os
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import ast,hashlib,json,subprocess,sys
from pathlib import Path
from datetime import datetime,timedelta,timezone
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;DATA=ROOT/'runs/historical_direct_query_inventory_20261109';MSK=timezone(timedelta(hours=3));STOP=datetime(2026,11,9,6,35,tzinfo=MSK)
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
remaining=(STOP-datetime.now(MSK)).total_seconds()
if remaining<=0:raise SystemExit('Save/stop reached')
guard_entrypoint(timeout=min(1200,remaining))
import numpy as np
from tracking.observed_spatial_reader import ObservedSpatialReader
from tracking.z13_readout import HexGeometry
from scripts.z13_target_model import receptor_features_rich,neighbor_aug
OTHER=[0,1,2,3]+list(range(12,34));SELECT=[0,5,11,16,17,22,28,33];NAMES=['C34','FlyVis','R134_temporal','R134_spatial','input8_temporal','center8','neighbor8']
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def cast(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,dict):return {k:cast(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [cast(x) for x in v]
    return v
def write(n,v):
    p=OUT/n;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f:json.dump(cast(v),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def verify(p):
    for name,h in p['sources'].items():assert sha(ROOT/name)==h,name
def stamp():return datetime.now(MSK).isoformat()
def seqkey(v,s):return hashlib.sha256(json.dumps([v,s],separators=(',',':')).encode()).hexdigest()[:32]
def journal(title,body):
    p=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';s=p.read_text(encoding='utf8');now=datetime.now(MSK);p.write_text(s.replace('## Журнал изменений\n',f'## Журнал изменений\n\n**{now:%d.%m.%Y, %H:%M} MSK** — {title}.\n\n{body}\n\n',1),encoding='utf8')
def register():
    inv=read(DATA/'results.json');source=read(DATA/'protocol.json');verify(source);assert inv['readiness_gate'] and len(inv['selected'])==12
    files=[Path(__file__),DATA/'protocol.json',DATA/'results.json',ROOT/'runs/z13_local_peak_20261021/geometry.json',ROOT/'tracking/observed_spatial_reader.py',ROOT/'tracking/hex_linear_interpolation.py',ROOT/'tracking/hex_background_filter.py',ROOT/'tracking/channel_balanced_correspondence.py',ROOT/'tracking/z13_readout.py',ROOT/'scripts/z13_target_model.py',ROOT/'tracking/flyvis_adapter.py',ROOT/'tracking/flyvis_activity.py',ROOT/'tracking/flyvis_motion.py',ROOT/'tracking/flyvis_checkpoint.py',ROOT/'runs/flyvis/results/flow/0000/000/best_chkpt',ROOT/'runs/flyvis/results/flow/0000/000/_meta.yaml',ROOT/'runs/z13_target_center_dev_20261018/C/normalization.json',ROOT/'runs/z13_target_center_dev_20261018/R1/normalization.json']+list((ROOT/'tools/flyvis-runtime/flyvis/network').glob('*.py'));requests={};cases=inv['selected']
    for c in cases:
        c['keys']=[]
        for seq in [c['sequence'][:c['reference_relative']+1],c['sequence']]:
            key=seqkey(c['video'],seq);c['keys'].append(key);requests[key]=dict(video=c['video'],sequence=seq,ready={})
        files.append(ROOT/source['videos'][c['video']]['data'])
    for key,r in requests.items():
        paths=sorted((ROOT/'runs').glob(f'*/maps/{key}.npz'))+sorted((ROOT/'runs').glob(f'*/features/{key}.npz'))+sorted((ROOT/'runs').glob(f'*/{key}.npz'))
        for path in paths:
            with np.load(path,allow_pickle=False) as z:
                for name in ['C34','C','C_OTHER','R1']:
                    if name not in z.files:continue
                    a=z[name]
                    if a.shape not in [(721,8),(721,26),(721,34)]:continue
                    slot='C8' if name=='C' and a.shape[1]==8 else ('C34' if name=='C' else name)
                    if slot not in r['ready']:r['ready'][slot]=dict(path=path.relative_to(ROOT).as_posix(),array=name);files.append(path)
    p=dict(registered=stamp(),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),deadline='2026-11-09T06:40:00+03:00',scientific_save_stop=STOP.isoformat(),cases=cases,requests=requests,
        frozen='Exactly12 first-clean queries from12 historical fit84 videos; one directly GT-confirmed own and one foreign box in the SAME current map/eye. Different videos from previous20/26 reader diagnostics, but historically used in older learned methods. No claim of a new dataset split.',
        method='Unchanged fixed19 offsets, support>=3 at all offsets, raw visible-edge correction only for image clipping, equal spatial correlation per reference-active channel, exact-constant interpolation repair. Snapshot reference and current from ONE archived prefix, same fixed2steps/dt1/60/fade1s, no clock change. Reference selection was evaluation GT-timestamp construction, not a deployable memory rule.',
        controls='C34 primary, C8 T4/T5, matched34temporal and34spatial input controls, input8temporal, center8 and neighbor8. Same retinal input sequences, sampled cells and query coordinates PER CANDIDATE. Do not intersect supports across candidates, which can force identical scores. Original saved native cost per candidate is the geometry-only local-rank control; actual original row assignment was NONE for all12. Local cost preference is NOT actual native ID preservation.',
        reuse='fr_hex already archived; no video/retina/detector/tracker replay. Read any exact ready C34/C8/OTHER/R1 maps. Recover only missing snapshot data, ready C8 witness<=1e-4, shared prefix propagated once. No threshold, mask or channel search; no use of GT/IDs by feature model or score.',
        metrics='Primary within-query own score>foreign AFTER maximum over19, tie=0.5 and abstention separately. Before search offset9. Record all scores/supports and native owners. Paired5000 video bootstrap seed42 C34 correctness minus saved negative cost and both34-channel controls; no pooled cross-query AUROC primary.',
        acceptance=[dict(criterion='All source/data/weights stable, two direct current identities and reference independently verified, all12 same-query maps/controls exact, independent interpolation/scalar correlations agree<=1e-9.',met=None),dict(criterion='C34 after search has positive paired95%video lower difference versus original local cost AND both matched34-channel image controls, with all refusals and ties retained.',met=None)],
        stop='If fixed test fails do not retune on these12 or replace primary by a favorable control. No ID/cost intervention or full HOTA claim. A passing reader gate still needs a causal policy, global solver and actual ID fate test.',sources={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(files))})
    write('protocol.json',p)
    path=ROOT/'experiments/project_history.json';cat=read(path);assert not any(c['id']==OUT.name for c in cat['experiments']);cat['experiments'].append(dict(id=OUT.name,title='Выбор своей из двух машин в одном запросе — FlyVis против входа и стоимости',area='flyvis',state='running',started='2026-11-09',question='Ставит ли зафиксированный пространственный считыватель свою машину выше чужой в одном прямо размеченном запросе до общего назначения?',hypothesis='Пространственные карты могут различать экземпляры при одинаковом текущемконтексте; сходство может также отражать общийфон и геометрию. Польза FlyVisне предполагается.',expected='12запросов/12другихвидео, одинаковыеC34/C8/обычныепризнаки и исходнаястоимость, безподборапорогов и повтора готовыхкарт.',acceptance=p['acceptance'],evidence=[dict(kind='file',ref=f'runs/{OUT.name}/protocol.json',claim='Правила и критерии до новых признаков и оценок')]))
    path.write_text(json.dumps(cat,ensure_ascii=False,indent=2)+'\n',encoding='utf8');journal('Зафиксировано прямое сравнение своей и чужой машины внутри одного запроса',f'*Постановка (`{OUT.name}`).* Послеаудита84историческихвидеоотобраны18чистыхзапросовна15видео; по предварительномуправилу выбраныпервые12UIDразныхвидео. Оба текущихкандидатаипрошлаясырaярамкаимеютоднозначнуюпрямуюGT. Все входыглазаготовы. Тотжеисправленныйсчитыватель,19сдвигов, семьпредставлений и исходнаястоимостьбезFlyVis. Основноечисло — правильныйвыбор внутрикаждойпары, недоляID. Вштатномпервомназначениистарымрядамне назначеноничего; локальныйпорядокстоимостейневыдаётсязаответобщегорешателя. Нетобучения/порогов/новогодетектора/повторакарт. До06:35сохранение/остановка. Источник `runs/{OUT.name}/protocol.json`.');print('Registered12 within-query comparisons',flush=True)
def arrays(key,p):
    r=p['requests'][key];a={}
    for name,item in r['ready'].items():
        with np.load(ROOT/item['path'],allow_pickle=False) as z:a[name]=z[item['array']].copy()
    path=OUT/'maps'/(key+'.npz')
    if path.with_suffix('.json').exists():
        assert sha(path)==read(path.with_suffix('.json'))['sha256']
        with np.load(path,allow_pickle=False) as z:
            for n in z.files:assert n not in a;a[n]=z[n].copy()
    if 'C34' not in a and 'C8' in a and 'C_OTHER' in a:
        c=np.empty((721,34),np.float32);c[:,4:12]=a['C8'];c[:,OTHER]=a['C_OTHER'];a['C34']=c
    return a
def restore():
    p=read(OUT/'protocol.json');verify(p)
    if (OUT/'maps_complete.json').exists():verify(read(OUT/'maps_complete.json'));return
    import torch,psutil
    from tracking.gpu_lock import GpuLock
    from tracking.flyvis_adapter import activate_flyvis
    from scripts.z13_target_model import build_arm
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
    source=read(DATA/'protocol.json');stats={n:read(ROOT/f'runs/z13_target_center_dev_20261018/{n}/normalization.json') for n in ['C','R1']};cm,cs=[np.array(stats['C'][n])[:34] for n in ['mean','sd']];rm,rs=[np.array(stats['R1'][n])[:34] for n in ['mean','sd']];bb=None;weights=None;files={};parity=[];new=0;(OUT/'maps').mkdir(exist_ok=True)
    with GpuLock('flyvis_fovea_gpu'),torch.no_grad():
        for i,c in enumerate(p['cases']):
            pending=[k for k in c['keys'] if 'C34' not in arrays(k,p) or 'R1' not in arrays(k,p)]
            if not pending:continue
            with np.load(ROOT/source['videos'][c['video']]['data'],allow_pickle=False) as z:
                off=int(z['ep_offset'][c['episode']]);hx=z['fr_hex'][off:off+c['current_relative']+1].astype(np.float32);assert np.array_equal(z['fr_frame'][off:off+len(hx)],[v[0] for v in c['sequence']]);assert np.array_equal(z['fr_crop'][off:off+len(hx)],[v[1] for v in c['sequence']])
            rich=receptor_features_rich(hx[None].astype(float));needcore=any('C34' not in arrays(k,p) for k in pending)
            if needcore:
                if bb is None:activate_flyvis(device='cuda');_,_,bb=build_arm('C',torch,None);assert bb.dev.type=='cuda' and not bb.net.training and bb.steps==2 and abs(bb.dt-1/60)<1e-12;weights={k:v.detach().clone() for k,v in bb.net.named_parameters()}
                x=torch.from_numpy(hx[None]).to(bb.dev);state=bb.init(x[:,:1])
            ends={len(p['requests'][k]['sequence'])-1:k for k in pending}
            for t in range(1,max(ends)+1):
                if needcore:state,feat=bb.step(state,x[:,t-1],x[:,t])
                if t not in ends:continue
                k=ends[t];old=arrays(k,p);values={};errors={}
                if 'R1' not in old:values['R1']=((rich[t][0].T-rm)/rs).astype(np.float32)
                if 'C34' not in old:
                    full=((feat[0].cpu().numpy().T.astype(float)-cm)/cs).astype(np.float32)
                    if 'C8' in old:errors['C8']=float(np.max(abs(full[:,4:12]-old['C8'])));assert errors['C8']<=1e-4;values['C_OTHER']=full[:,OTHER]
                    else:values['C34']=full
                dest=OUT/'maps'/(k+'.npz');assert not dest.exists()
                with dest.open('xb') as out:np.savez_compressed(out,**values)
                write(Path('maps')/(k+'.json'),dict(sha256=sha(dest),errors=errors,protocol_sha256=sha(OUT/'protocol.json')));files[dest.relative_to(ROOT).as_posix()]=sha(dest);new+=1;parity.append(errors)
            if needcore:del x,state,feat
            (OUT/'status.json').write_text(json.dumps(dict(updated=stamp(),stage='restore',done=i+1,total=12,new_maps=new)),encoding='utf8');print('MAPS',i+1,12,flush=True)
        if bb is not None:assert all(torch.equal(v,weights[k]) for k,v in bb.net.named_parameters())
    for key in p['requests']:
        a=arrays(key,p);assert 'C34' in a and 'R1' in a
        for item in p['requests'][key]['ready'].values():files[item['path']]=sha(ROOT/item['path'])
    verify(p);write('maps_complete.json',dict(passed=True,sources=files,new_snapshot_files=new,total_snapshots=len(p['requests']),weights_unchanged=True,no_new_retinal_views=True,parity=parity));print('Restored missing snapshots only',flush=True)
def compare():
    from scipy.interpolate import LinearNDInterpolator
    p=read(OUT/'protocol.json');verify(p);verify(read(OUT/'maps_complete.json'))
    if (OUT/'results.json').exists():raise SystemExit('Ready comparison exists')
    g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');reader=ObservedSpatialReader(g['points'],g['eye_size'],[1280,720]);nb=HexGeometry(np.array(g['points']),g['eye_size']).neighbors;records=[];error=0.
    def features(k):
        a=arrays(k,p);c=a['C34'].astype(float);r=a['R1'].astype(float);cen=r[:,SELECT[:4]]
        return dict(C34=c,FlyVis=c[:,4:12],R134_temporal=r,R134_spatial=neighbor_aug(r[:,:17].T[None],nb)[0].T,input8_temporal=r[:,SELECT],center8=np.concatenate([cen,cen],1),neighbor8=neighbor_aug(cen.T[None],nb)[0].T)
    for c in p['cases']:
        ref,cur=[features(k) for k in c['keys']];variants={}
        for name in NAMES:
            scores=[]
            for candidate in c['candidates']:
                plan=candidate['plan'];q=reader.compare(ref[name],cur[name],plan);idx=np.array(plan['cells']);template=ref[name][idx];patches=LinearNDInterpolator(reader.mesh,cur[name])(np.array(plan['query'])).transpose(1,0,2);values=[]
                # Independent mathematical expression, before normalization.
                # Exact-constant channels are identified from triangle vertices;
                # ordinary SciPy sums are otherwise only a roundoff witness.
                for patch in patches:
                    a=template-template.mean(0);b=patch-patch.mean(0);aa=np.ptp(template,axis=0)!=0;bb=np.ptp(patch,axis=0)!=0;a[:,~aa]=0;b[:,~bb]=0;an=np.linalg.norm(a,axis=0);bn=np.linalg.norm(b,axis=0);den=an*bn;rho=np.divide((a*b).sum(0),den,out=np.zeros_like(an),where=den>0);values.append(float(rho.sum()/aa.sum()) if aa.any() and bb.any() else None)
                # Reference/current maps here are genuinely varying, but if a
                # SciPy roundoff witness is degenerate, do not override the fixed
                # mathematically correct sampler or invent a tolerance gate.
                for a,b in zip(q['scores'],values):
                    if a is not None and b is not None:error=max(error,abs(a-b))
                scores.append(dict(kind=candidate['kind'],raw_id=candidate['raw_id'],cells=plan['cells'],scores=q['scores'],fixed=q['scores'][9],maximum=q['score'],winner=q['winner']))
            variants[name]=scores
        records.append(dict(video=c['video'],uid=c['uid'],frame=c['frame'],tid=c['tid'],reference_frame=c['reference_frame'],variants=variants,original_cost=[dict(kind=q['kind'],cost=q['original_cost']) for q in c['candidates']],current_native_owners=[q['current_native_owners'] for q in c['candidates']]));print('COMPARED',len(records),12,flush=True)
    verify(p);write('results.json',dict(passed=True,records=records,independent_scipy_scalar_max_error=error,geometry_and_inputs_shared=True,zero_threshold_tuning=True,zero_id_assignment=True));print('Within-query comparison complete',error,flush=True)
if __name__=='__main__':{'register':register,'restore':restore,'compare':compare}[sys.argv[1]]()
