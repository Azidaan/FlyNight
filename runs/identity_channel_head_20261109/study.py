"""One preregistered learned reader, with causal data and matched controls."""
import os
import sys
import json
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime, timezone, timedelta

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DATA = ROOT/'runs/identity_channel_training_readiness_20261109'
MSK = timezone(timedelta(hours=3))
STOP = datetime(2026,11,9,23,54,tzinfo=MSK)
if len(sys.argv) > 1 and sys.argv[1] != 'restore':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '2'
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
if not (OUT/'status.json').exists():
    (OUT/'status.json').write_text(json.dumps(dict(stage='starting',updated=datetime.now(MSK).isoformat())),encoding='utf8')
remaining = (STOP-datetime.now(MSK)).total_seconds()
assert remaining > 0
guard_entrypoint(timeout=min(3600,remaining))
import numpy as np
import psutil
psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
from tracking.observed_spatial_reader import ObservedSpatialReader
from tracking.channel_identity_head import (spatial_channel_features, causal_geometry,
    geometry_transform, fit_ridge_logistic, decision_function)

OTHER = [0,1,2,3]+list(range(12,34))
SELECT = [0,5,11,16,17,22,28,33]
NAMES = ['C34','R134_spatial','R134_temporal','neighbor8']

def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def cast(value):
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,dict): return {k:cast(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return [cast(v) for v in value]
    return value

def write(name,obj):
    path = OUT/name
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf8') as f:
        json.dump(cast(obj),f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n')

def verify(protocol):
    for path,value in protocol['sources'].items():
        assert sha(ROOT/path) == value,path

def status(stage,**kwargs):
    (OUT/'status.json').write_text(json.dumps(dict(stage=stage,updated=datetime.now(MSK).isoformat(),**kwargs)),encoding='utf8')

def journal(title,body):
    path = ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md'
    text = path.read_text(encoding='utf8')
    stamp = datetime.now(MSK).strftime('%d.%m.%Y, %H:%M MSK')
    path.write_text(text.replace('## Журнал изменений\n','## Журнал изменений\n\n**'+stamp+'** — '+title+'\n\n'+body+'\n\n',1),encoding='utf8')

def seqkey(video,sequence):
    return hashlib.sha256(json.dumps([video,sequence],separators=(',',':')).encode()).hexdigest()[:32]

def register():
    assert not (OUT/'protocol.json').exists()
    assert read(DATA/'audit.json')['passed'] and read(DATA/'final_checks.json')['passed']
    old = read(DATA/'protocol.json')
    edges = [e for e in read(DATA/'results.json')['edges'] if e['cohort']=='strict_last' and e['supported']]
    assert len(edges)==91 and sum(e['label']=='own' for e in edges)==15
    cases = {}
    requests = {}
    files = [Path(__file__), ROOT/'tracking/channel_identity_head.py',ROOT/'tests/test_channel_identity_head.py',
        DATA/'results.json',DATA/'protocol.json',DATA/'audit.json',DATA/'causal_reference_audit.json',
        ROOT/'runs/z13_local_peak_20261021/geometry.json',ROOT/'tracking/hex_linear_interpolation.py',
        ROOT/'tracking/channel_balanced_correspondence.py',ROOT/'tracking/observed_spatial_reader.py',
        ROOT/'tracking/z13_readout.py',ROOT/'tracking/foveated_eyes.py',ROOT/'tracking/flyvis_adapter.py',
        ROOT/'tracking/flyvis_motion.py',ROOT/'tracking/flyvis_activity.py',ROOT/'tracking/flyvis_checkpoint.py',
        ROOT/'scripts/z13_target_model.py',ROOT/'scripts/run_bdd_video_rate_experiment.py',
        ROOT/'runs/flyvis/results/flow/0000/000/best_chkpt',ROOT/'runs/flyvis/results/flow/0000/000/_meta.yaml',
        ROOT/'runs/z13_target_center_dev_20261018/C/normalization.json',
        ROOT/'runs/z13_target_center_dev_20261018/R1/normalization.json']
    files += list((ROOT/'tools/flyvis-runtime/flyvis/network').glob('*.py'))
    for e in edges:
        pair=(e['video'],e['uid'])
        sequences=[e['sequence'][:e['reference_relative']+1],e['sequence']]
        keys=[seqkey(e['video'],s) for s in sequences]
        if pair not in cases:
            cases[pair]=dict(video=e['video'],uid=e['uid'],episode=e['episode'],keys=keys,
                current_relative=len(e['sequence'])-1,sequence=e['sequence'])
        assert cases[pair]['keys']==keys and e['reference_relative']==3
        e['keys']=keys
        for key,seq in zip(keys,sequences):
            request=dict(video=e['video'],sequence=seq,ready={})
            if key in requests: assert requests[key]==request
            requests[key]=request
    videos={v:old['videos'][v] for v in sorted({e['video'] for e in edges})}
    for v,info in videos.items():
        path=ROOT/info['data']
        assert sha(path)==old['sources'][info['data']]
        files.append(path)
    source_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in files}
    ignored=[]
    # Reuse only caches whose own passport binds the same input, checkpoint,
    # feature implementation and normalization. A key alone is not provenance.
    candidates=[]
    for directory in sorted((ROOT/'runs').iterdir()):
        if not directory.is_dir() or directory==OUT: continue
        for sub in ['maps','features']:
            folder=directory/sub
            if folder.is_dir():
                candidates += [p for p in folder.glob('*.npz') if p.stem in requests]
    common=['scripts/z13_target_model.py','runs/z13_target_center_dev_20261018/R1/normalization.json']
    core=['runs/flyvis/results/flow/0000/000/best_chkpt','runs/z13_target_center_dev_20261018/C/normalization.json']
    for path in sorted(candidates):
        meta=path.with_suffix('.json');passport=path.parent.parent/'protocol.json'
        if not meta.exists() or not passport.exists():
            ignored.append(dict(path=path.relative_to(ROOT).as_posix(),reason='no_map_passport'));continue
        m=read(meta);pp=read(passport);ss=pp.get('sources',{})
        if 'sha256' not in m:
            ignored.append(dict(path=path.relative_to(ROOT).as_posix(),reason='map_receipt_schema_not_yet_verified'));continue
        if m.get('sha256')!=sha(path) or (m.get('protocol_sha256') and m['protocol_sha256']!=sha(passport)):
            raise AssertionError('Existing map receipt mismatch: '+str(path))
        key=path.stem;video=requests[key]['video'];required=common+[videos[video]['data']]
        if any(ss.get(k)!=source_hashes.get(k) for k in required):
            ignored.append(dict(path=path.relative_to(ROOT).as_posix(),reason='input_or_feature_passport_not_matched'));continue
        with np.load(path,allow_pickle=False) as z:
            for name in ['C34','C8','C','C_OTHER','R1']:
                if name not in z.files:continue
                a=z[name]
                if a.shape not in [(721,8),(721,26),(721,34)]:continue
                slot='C8' if name=='C' and a.shape[1]==8 else ('C34' if name=='C' else name)
                if slot!='R1' and any(ss.get(k)!=source_hashes.get(k) for k in core):
                    ignored.append(dict(path=path.relative_to(ROOT).as_posix(),reason='core_passport_not_matched'));continue
                if slot not in requests[key]['ready']:
                    requests[key]['ready'][slot]=dict(path=path.relative_to(ROOT).as_posix(),array=name)
                    source_hashes[path.relative_to(ROOT).as_posix()]=sha(path)
                    source_hashes[meta.relative_to(ROOT).as_posix()]=sha(meta)
                    source_hashes[passport.relative_to(ROOT).as_posix()]=sha(passport)
    protocol=dict(registered=datetime.now(MSK).isoformat(),head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        authorization='research_identity_synthesis_20261109/authorization.json',save_stop=STOP.isoformat(),hard_stop='2026-11-09T23:59:00+03:00',
        action='Replace equal channel averaging by one fixed regularized learned decision on per-channel spatial correlations; geometry and input-image models use the same procedure. No core training.',
        diagnostic_number=2,previous_diagnostic='runs/identity_channel_training_readiness_20261109/audit.json',maximum_diagnostics=3,
        edges=edges,cases=list(cases.values()),requests=requests,videos=videos,sources=source_hashes,ignored_unproven_caches=ignored,
        sample='Exactly 91 supported strict-last edges, 15 own/76 foreign, 32 videos, excluding all 15 prior directly inspected videos. Earlier-labelled 204 edges excluded. Historical fit84 is not a new official split.',
        spatial='Use already computed single-candidate support and 19 fixed offsets; no alternative candidate participates. Spatial Pearson correlation per channel; exact constant handling; choose shift by the unchanged equal-channel reader, first maximum tie rule. Features rho_c*C/number_reference_active; their mean equals the old scalar. No spatial/channel/scale threshold fitted.',
        geometry=['original_cost','frame_gap','log_width_ratio','log_height_ratio','center_dx_over_reference_width','center_dy_over_reference_height','log1p_cell_count'],
        representations=NAMES,models=['G']+[f'G_{name}_{kind}' for name in NAMES for kind in ['mean','channels']],
        model='Class-balanced mean logistic loss + ||w||^2/2, lambda=1 fixed; free intercept. Deterministic damped Newton, zero init, at most100 iterations, gradient tolerance1e-7, target1e-9. Strict decision z>0, no probability-calibration claim. Geometry standardized on training videos only; correlation features not standardized.',
        validation='Leave one complete video out,32 fixed folds. Train valid feature rows only; if a feature training fold loses a class, use the separately fitted G for the entire fold. Undefined feature on a test edge also uses that G prediction. Never put GT/track IDs, video IDs or candidate owners into features.',
        uncertainty='Paired bootstrap of32 video clusters with replacement,4000 draws,seed20261109; percentile95% intervals. Report own accepted/rejected and foreign accepted/rejected, balanced accuracy, AUROC as descriptive only. No model selected by test threshold.',
        pilot_gate='For a channel model to enter a bounded actual-ID pilot, balanced accuracy must strictly exceed G and its own scalar model, foreign accepts must be no greater than both, and own accepts must strictly exceed both. This gate is admission only, not proven benefit. All representations are reported; no retuning failed models. If several pass, select highest balanced accuracy, ties lower foreign accepts, then fewer channels, then protocol order. This selection is disclosed as development selection.',
        flyvis_claim='An additional FlyVis contribution requires paired confidence intervals against both matched34-channel image controls and G, then actual independent ID/HOTA evidence; passing a point gate does not establish it.',
        failure='If no channel model passes the point gate, stop this learned-channel version, preserve unfavorable results and do not tune lambda, thresholds, channels, shifts or samples after inspection.',
        literature=[dict(url='https://www.jmlr.org/papers/volume9/lin08b/lin08b.pdf',section='1, equations1-2',use='Regularized logistic objective; our class weighting and free unpenalized intercept are explicit experimental choices, not the paper exact implementation.'),dict(url='https://link.springer.com/article/10.1186/s12874-016-0267-3',use='Small sample caution: no universal EPV adequacy guarantee.')])
    write('protocol.json',protocol)
    catpath=ROOT/'experiments/project_history.json';catalog=read(catpath)
    assert not any(c['id']==OUT.name for c in catalog['experiments'])
    catalog['experiments'].append(dict(id=OUT.name,title='Принадлежность рамки — обучаемые совпадения каналов',area='flyvis',state='running',started='2026-11-09',
        question='Помогают ли отдельные пространственные совпадения каналов отвергать чужую рамку лучше геометрии и равного среднего?',
        hypothesis='Равное усреднение может терять полезное различие каналов. Ограниченная обучаемая оценка может сохранить его; сильнейшая альтернатива — тот же эффект обычных пространственно-временных признаков или переобучение на сюжет.',
        expected='Одна межвидеовая проверка зафиксированного решения с одинаковыми контролями, затем реальный пилот назначения только при прохождении критерия.',
        acceptance=[dict(criterion='Математика, совпадение среднего с прежним считывателем, отсутствие утечки видео и неизменность источников подтверждены.',met=None),dict(criterion=protocol['pilot_gate'],met=None)],
        obtained='',interpretation='',decision='',cannot_claim='',evidence=[dict(kind='file',ref='runs/'+OUT.name+'/protocol.json',claim='Условия зарегистрированы до извлечения признаков и обучения.')],tags=['fly-problem-P1']))
    temp=catpath.with_suffix('.head.tmp');temp.write_text(json.dumps(catalog,ensure_ascii=False,indent=2)+'\n',encoding='utf8');temp.replace(catpath)
    glossary_path=ROOT/'experiments/metrics_glossary.json';glossary=read(glossary_path)
    glossary['metrics'].setdefault('candidate_identity_balanced_accuracy',dict(title='Сбалансированная точность принадлежности рамки',ru='Половина суммы доли принятых своих и доли отвергнутых чужих связей при заранее фиксированном нулевом пороге модели.',better='higher',unit='доля',range='0..1',tool='tracking/channel_identity_head.py',caveat='Диагностика отдельных рёбер с исключением видео. Не доказывает сохранение ID, отсутствие вреда конкурентам или прибавку HOTA.',source='runs/'+OUT.name+'/protocol.json'))
    glossary_path.write_text(json.dumps(glossary,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    journal('Зафиксирована одна обучаемая оценка совпадений каналов',f'*Постановка (`{OUT.name}`).* Вместо равного среднего проверяется регуляризованная линейная оценка отдельных пространственных совпадений. Готовые 91 пара (15 своих и 76 чужих), 32 видео, только последнее фактическое наблюдение; источники `runs/identity_channel_training_readiness_20261109/audit.json`. Те же поддержка, 19 сдвигов и выбор сдвига прежним считывателем. Геометрический контроль и одинаковые обучаемые модели на двух 34-канальных входных представлениях и восьми соседних признаках. Исключение целого видео, штраф 1 и порог 0 фиксированы. Критерий допуска к реальным ID, отказ от неудачной версии и формулы: `runs/{OUT.name}/protocol.json`. Это вторая диагностика одной правки; ядро не обучается. Литературная основа — Lin, Weng, Keerthi (2008), JMLR9:627–650, раздел1, формула2; наши балансировка и свободный сдвиг указаны отдельно.')
    status('registered',edges=len(edges),videos=len(videos),snapshots=len(requests),cases=len(cases))
    print('Registered',len(edges),'edges;',len(cases),'queries;',len(requests),'snapshots',flush=True)

def mathematics():
    p=read(OUT/'protocol.json');verify(p)
    import pytest
    assert pytest.main(['tests/test_channel_identity_head.py','-q'])==0
    write('math_checks.json',dict(passed=True,tests=8,finite_difference_gradient_hessian=True,finite_separable_fit=True,
        old_scalar_and_shift_identity=True,constant_patch_abstention=True,geometry_train_only_normalization=True,
        no_real_scores_or_training=True,sources={k:v for k,v in p['sources'].items() if k.startswith(('tracking/','tests/'))}))
    print('Mathematical implementation gate passed',flush=True)

def arrays(key,p):
    values={}
    for name,meta in p['requests'][key]['ready'].items():
        with np.load(ROOT/meta['path'],allow_pickle=False) as z:values[name]=z[meta['array']].copy()
    path=OUT/'maps'/(key+'.npz')
    if path.with_suffix('.json').exists():
        assert sha(path)==read(path.with_suffix('.json'))['sha256']
        with np.load(path,allow_pickle=False) as z:
            for name in z.files:assert name not in values;values[name]=z[name].copy()
    if 'C34' not in values and 'C8' in values and 'C_OTHER' in values:
        c=np.empty((721,34),np.float32);c[:,4:12]=values['C8'];c[:,OTHER]=values['C_OTHER'];values['C34']=c
    return values

def restore():
    p=read(OUT/'protocol.json');verify(p);assert read(OUT/'math_checks.json')['passed']
    if (OUT/'maps_complete.json').exists():verify(read(OUT/'maps_complete.json'));return
    import torch
    from tracking.gpu_lock import GpuLock
    from tracking.flyvis_adapter import activate_flyvis
    from scripts.z13_target_model import build_arm,receptor_features_rich
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    stats={n:read(ROOT/f'runs/z13_target_center_dev_20261018/{n}/normalization.json') for n in ['C','R1']}
    cm,cs=[np.array(stats['C'][n])[:34] for n in ['mean','sd']]
    rm,rs=[np.array(stats['R1'][n])[:34] for n in ['mean','sd']]
    bb=None;weights=None;new=0;parity=[];files={};(OUT/'maps').mkdir(exist_ok=True)
    with GpuLock('flyvis_fovea_gpu'),torch.no_grad():
        for i,case in enumerate(p['cases']):
            pending=[k for k in case['keys'] if 'C34' not in arrays(k,p) or 'R1' not in arrays(k,p)]
            if not pending:continue
            with np.load(ROOT/p['videos'][case['video']]['data'],allow_pickle=False) as z:
                off=int(z['ep_offset'][case['episode']]);n=case['current_relative']+1
                hx=z['fr_hex'][off:off+n].astype(np.float32)
                assert np.array_equal(z['fr_frame'][off:off+n],[v[0] for v in case['sequence']])
                assert np.array_equal(z['fr_crop'][off:off+n],[v[1] for v in case['sequence']])
            rich=receptor_features_rich(hx[None].astype(float))
            needcore=any('C34' not in arrays(k,p) for k in pending)
            if needcore:
                if bb is None:
                    activate_flyvis(device='cuda');_,_,bb=build_arm('C',torch,None)
                    assert bb.dev.type=='cuda' and not bb.net.training and bb.steps==2 and abs(bb.dt-1/60)<1e-12
                    weights={k:v.detach().clone() for k,v in bb.net.named_parameters()}
                x=torch.from_numpy(hx[None]).to(bb.dev);state=bb.init(x[:,:1])
            ends={len(p['requests'][k]['sequence'])-1:k for k in pending}
            for t in range(1,max(ends)+1):
                if needcore:state,feat=bb.step(state,x[:,t-1],x[:,t])
                if t not in ends:continue
                key=ends[t];old=arrays(key,p);values={};errors={}
                if 'R1' not in old:values['R1']=((rich[t][0].T-rm)/rs).astype(np.float32)
                if 'C34' not in old:
                    full=((feat[0].cpu().numpy().T.astype(float)-cm)/cs).astype(np.float32)
                    if 'C8' in old:
                        errors['C8']=float(np.max(abs(full[:,4:12]-old['C8'])));assert errors['C8']<=1e-4
                        values['C_OTHER']=full[:,OTHER]
                    else:values['C34']=full
                dest=OUT/'maps'/(key+'.npz');assert not dest.exists()
                with dest.open('xb') as f:np.savez_compressed(f,**values)
                write(Path('maps')/(key+'.json'),dict(sha256=sha(dest),errors=errors,protocol_sha256=sha(OUT/'protocol.json')))
                files[dest.relative_to(ROOT).as_posix()]=sha(dest);new+=1;parity.append(errors)
            if needcore:del x,state,feat
            status('restore',done=i+1,total=len(p['cases']),new_snapshots=new)
            print('MAPS',i+1,len(p['cases']),new,flush=True)
        if bb is not None:assert all(torch.equal(v,weights[k]) for k,v in bb.net.named_parameters())
    for key in p['requests']:
        a=arrays(key,p);assert 'C34' in a and 'R1' in a
        for item in p['requests'][key]['ready'].values():files[item['path']]=sha(ROOT/item['path'])
        path=OUT/'maps'/(key+'.npz')
        if path.exists():files[path.relative_to(ROOT).as_posix()]=sha(path)
    verify(p)
    write('maps_complete.json',dict(passed=True,sources=files,new_snapshot_files=new,total_snapshots=len(p['requests']),weights_unchanged=True,no_new_retinal_views=True,parity=parity))
    print('Only missing snapshot arrays restored',flush=True)

def extract():
    p=read(OUT/'protocol.json');verify(p);verify(read(OUT/'maps_complete.json'))
    from tracking.z13_readout import HexGeometry
    from scripts.z13_target_model import neighbor_aug
    g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');reader=ObservedSpatialReader(g['points'],g['eye_size'],[1280,720]);nb=HexGeometry(np.array(g['points']),g['eye_size']).neighbors
    cache={};result=[];parity=0.
    def features(key):
        if key not in cache:
            a=arrays(key,p);c=a['C34'].astype(float);r=a['R1'].astype(float);cen=r[:,SELECT[:4]]
            cache[key]=dict(C34=c,R134_temporal=r,R134_spatial=neighbor_aug(r[:,:17].T[None],nb)[0].T,neighbor8=neighbor_aug(cen.T[None],nb)[0].T)
        return cache[key]
    for i,e in enumerate(p['edges']):
        ref,cur=[features(k) for k in e['keys']];variants={};plan=dict(cells=e['cells'],query=e['query'])
        for name in NAMES:
            values=spatial_channel_features(reader.mesh,ref[name],cur[name],e['cells'],e['query'])
            previous=reader.compare(ref[name],cur[name],plan)
            assert values['winner']==previous['winner']
            if values['valid']:
                parity=max(parity,abs(values['scalar']-previous['score']),abs(np.mean(values['channels'])-previous['score']))
            assert values['scores']==previous['scores']
            variants[name]=values
        geom=causal_geometry(e['reference_box'],e['current_box'],e['original_cost'],e['frame']-e['sequence'][3][0],len(e['cells']))
        result.append(dict(video=e['video'],uid=e['uid'],old=e['old'],raw_id=e['raw_id'],label=e['label'],geometry=geom,variants=variants))
        print('FEATURES',i+1,len(p['edges']),flush=True)
    assert parity<=1e-12;verify(p)
    write('features.json',dict(passed=True,records=result,scalar_and_shift_parity_max_error=parity,one_candidate_support=True,labels_not_used_to_extract=True))
    print('Causal per-channel features complete',flush=True)

def measures(y,z):
    accepted=np.asarray(z)>0;own=np.asarray(y)==1
    tp=int((accepted&own).sum());fn=int((~accepted&own).sum());fp=int((accepted&~own).sum());tn=int((~accepted&~own).sum())
    ba=.5*(tp/(tp+fn)+tn/(tn+fp)) if tp+fn and tn+fp else None
    pos=np.asarray(z)[own];neg=np.asarray(z)[~own]
    auroc=float(((pos[:,None]>neg).sum()+.5*(pos[:,None]==neg).sum())/(len(pos)*len(neg))) if len(pos) and len(neg) else None
    return dict(own_accept=tp,own_reject=fn,foreign_accept=fp,foreign_reject=tn,balanced_accuracy=ba,auroc=auroc)

def fit():
    p=read(OUT/'protocol.json');verify(p);f=read(OUT/'features.json');assert f['passed']
    rows=f['records'];videos=sorted({r['video'] for r in rows});groups=np.array([r['video'] for r in rows]);y=np.array([r['label']=='own' for r in rows],int);G=np.array([r['geometry'] for r in rows]);scores={n:np.full(len(y),np.nan) for n in p['models']};fits=[]
    for fold,video in enumerate(videos):
        train=np.flatnonzero(groups!=video);test=np.flatnonzero(groups==video)
        assert not set(groups[train])&set(groups[test]) and set(y[train])=={0,1}
        gt,ge,stats=geometry_transform(G[train],G[test]);base=fit_ridge_logistic(gt,y[train]);gb=decision_function(base,ge);scores['G'][test]=gb
        fits.append(dict(held_out_video=video,model='G',training_videos=sorted(set(groups[train])),fit=base,geometry_normalization=stats))
        for name in NAMES:
            valid=np.array([r['variants'][name]['valid'] for r in rows]);tr=valid[train];te=valid[test]
            for kind in ['mean','channels']:
                model_name=f'G_{name}_{kind}';scores[model_name][test]=gb
                if not tr.any() or set(y[train][tr])!={0,1}:
                    fits.append(dict(held_out_video=video,model=model_name,fallback='training_feature_rows_lack_both_classes'));continue
                def extra(indices):
                    return np.array([[rows[j]['variants'][name]['scalar']] if kind=='mean' else rows[j]['variants'][name]['channels'] for j in indices],float)
                # Geometry normalization remains exactly the common G training-fold transform.
                xt=np.column_stack([gt[tr],extra(train[tr])]);model=fit_ridge_logistic(xt,y[train][tr])
                if te.any():scores[model_name][test[te]]=decision_function(model,np.column_stack([ge[te],extra(test[te])]))
                fits.append(dict(held_out_video=video,model=model_name,training_videos=sorted(set(groups[train][tr])),fit=model,test_fallbacks=int((~te).sum())))
        status('fit',done=fold+1,total=len(videos));print('FOLD',fold+1,len(videos),flush=True)
    assert all(np.isfinite(z).all() for z in scores.values())
    metrics={name:measures(y,z) for name,z in scores.items()}
    gates={}
    for name in NAMES:
        a=metrics[f'G_{name}_channels'];b=metrics[f'G_{name}_mean'];g=metrics['G']
        gates[name]=(a['balanced_accuracy']>max(b['balanced_accuracy'],g['balanced_accuracy'])
            and a['foreign_accept']<=min(b['foreign_accept'],g['foreign_accept'])
            and a['own_accept']>max(b['own_accept'],g['own_accept']))
    choices=[name for name in NAMES if gates[name]]
    selected=sorted(choices,key=lambda n:(-metrics[f'G_{n}_channels']['balanced_accuracy'],metrics[f'G_{n}_channels']['foreign_accept'],34 if n!='neighbor8' else 8,NAMES.index(n)))[0] if choices else None
    rng=np.random.default_rng(20261109);draws=[]
    byvideo=[np.flatnonzero(groups==v) for v in videos]
    for _ in range(4000):
        indices=np.concatenate([byvideo[k] for k in rng.integers(len(videos),size=len(videos))])
        if len(set(y[indices]))==2:draws.append(indices)
    intervals={}
    for name in NAMES:
        current=f'G_{name}_channels'
        for baseline in ['G',f'G_{name}_mean']+(['G_R134_spatial_channels','G_R134_temporal_channels'] if name=='C34' else []):
            values=[measures(y[i],scores[current][i])['balanced_accuracy']-measures(y[i],scores[baseline][i])['balanced_accuracy'] for i in draws]
            intervals[current+'__'+baseline]=dict(point=metrics[current]['balanced_accuracy']-metrics[baseline]['balanced_accuracy'],ci95=np.percentile(values,[2.5,97.5]),valid_draws=len(values))
    verify(p)
    write('predictions.json',dict(passed=True,records=[dict(video=r['video'],uid=r['uid'],old=r['old'],raw_id=r['raw_id'],label=r['label'],scores={name:z[i] for name,z in scores.items()}) for i,r in enumerate(rows)],folds=fits,zero_threshold_tuning=True,zero_ID_assignment=True))
    write('summary.json',dict(passed=True,edges=len(rows),own=int(y.sum()),foreign=int((1-y).sum()),videos=len(videos),metrics=metrics,point_pilot_gates=gates,selected_development_arm=selected,paired_video_intervals=intervals,scalar_parity_max_error=f['scalar_and_shift_parity_max_error'],new_core_training=False,new_ID_assignment=False,conclusion='Admission gate only; inspect paired uncertainty and then actual ID fate. No tracker or unique-core benefit is proved here.'))
    print(json.dumps(cast(dict(metrics=metrics,gates=gates,selected=selected)),ensure_ascii=False),flush=True)

if __name__=='__main__':
    {'register':register,'mathematics':mathematics,'restore':restore,'extract':extract,'fit':fit}[sys.argv[1]]()
