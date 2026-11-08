"""Frozen attribution and one parameter-free channel-normalization candidate."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import ast,hashlib,json,subprocess,sys
from pathlib import Path
from datetime import datetime,timedelta,timezone
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
BASE=ROOT/'runs/bbox_fit_maps_20261108';PARENT=ROOT/'runs/background_patch_gate_20261108';WIDE=ROOT/'runs/decoder34_identity_gate_20261108';FULL=ROOT/'runs/full_decoder_outputs_20261108';SEVEN=ROOT/'runs/singleton_support_size_20261108'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
guard_entrypoint(timeout=900)
import numpy as np
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
from tracking.channel_balanced_correspondence import channel_balanced_similarity,centered_contributions,channel_difference_attribution
from tracking.flyvis_activity import OUTPUT_CELL_TYPES
from tracking.z13_readout import HexGeometry
from scripts.z13_target_model import neighbor_aug,R1_TAUS
OTHER=[0,1,2,3]+list(range(12,34));SELECT=[0,5,11,16,17,22,28,33]
NAMES=['FlyVis','center8','neighbor8','input8_temporal','C34','R134_temporal','R134_spatial']
TARGETS=[('00982f76-83a093ef',156,88,144),('00d7268f-fd4487be',236,73,106)]
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cast(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,dict):return {k:cast(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [cast(x) for x in v]
    return v
def write(name,v):
    p=OUT/name;p.parent.mkdir(exist_ok=True)
    with p.open('x',encoding='utf8') as f:json.dump(cast(v),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def verify(p):
    for path,digest in p['sources'].items():assert sha(ROOT/path)==digest,path
def key(e):return e['video'],e.get('frame'),e.get('old_tid',e.get('old')),e.get('new_tid',e.get('new'))
def filename(e):return '_'.join(map(str,key(e)))+'.json'
def journal(title,text):
    path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';body=path.read_text(encoding='utf8');now=datetime.now(timezone(timedelta(hours=3)))
    entry=f'**{now:%d.%m.%Y, %H:%M} MSK** — {title}.\n\n'+text+'\n\n';path.write_text(body.replace('## Журнал изменений\n','## Журнал изменений\n\n'+entry,1),encoding='utf8')
def sampler():
    source=PARENT/'study.py';node=next(n for n in ast.parse(source.read_text(encoding='utf8')).body if isinstance(n,ast.FunctionDef) and n.name=='sample')
    ns=dict(np=np,LinearNDInterpolator=LinearNDInterpolator);exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns);return ns['sample']
def channels(name):
    temporal=[f'EMA tau={t:.6g}' for t in R1_TAUS]+[f'input-EMA tau={t:.6g}' for t in R1_TAUS]
    central=[temporal[i] for i in SELECT[:4]]
    return dict(FlyVis=list(OUTPUT_CELL_TYPES[4:12]),C34=list(OUTPUT_CELL_TYPES),center8=central+central,
        neighbor8=central+['neighbor '+n for n in central],input8_temporal=[temporal[i] for i in SELECT],
        R134_temporal=temporal,R134_spatial=temporal[:17]+['neighbor '+n for n in temporal[:17]])[name]
def register():
    assert read(OUT/'mathematical_checks.json')['passed'] and read(OUT/'delta_mathematical_checks.json')['passed']
    files=[Path(__file__),ROOT/'tracking/channel_balanced_correspondence.py',ROOT/'tests/test_channel_balanced_correspondence.py',OUT/'math_check.py',OUT/'mathematical_checks.json',OUT/'delta_math_check.py',OUT/'delta_mathematical_checks.json',
        PARENT/'protocol.json',PARENT/'study.py',PARENT/'results.json',WIDE/'results.json',FULL/'completed.json',ROOT/'runs/z13_local_peak_20261021/geometry.json',ROOT/'tracking/z13_readout.py',ROOT/'scripts/z13_target_model.py',ROOT/'tracking/flyvis_activity.py',
        ROOT/'runs/template_identity_audit_20261108/results.json',ROOT/'runs/template_identity_audit_20261108/counterexample.json',SEVEN/'protocol.json',SEVEN/'results.json',ROOT/'runs/singleton_forward_backward_transfer_20261108/maps_complete.json']
    old=read(PARENT/'protocol.json')
    for e in old['events']:
        files += [PARENT/'events'/filename(e),WIDE/'events'/filename(e)]
        for k in [e['reference_key'],e['current_key']]:files += [BASE/'features'/(k+'.npz'),FULL/'maps'/(k+'.npz')]
    files += [ROOT/p for p in read(ROOT/'runs/singleton_forward_backward_transfer_20261108/maps_complete.json')['sources']]
    p=dict(registered=datetime.now(timezone(timedelta(hours=3))).isoformat(),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),deadline='2026-11-09T07:40:00+03:00',
        requested='Attribute the wrong score order of88->144 and73->106 to cells/channels, then test one common reader correction with no-FlyVis controls.',
        math='Old S=sum_pc d_pc=sum_c w_c*rho_c. Exact symmetric two-factor difference distinguishes changed spatial correlation and changed relative channel amplitude weight. Algebraic attribution is not a causal neuron intervention.',
        intervention='Only replace shared Frobenius norm by equal per-channel spatial NCC, averaged over exactly nonconstant REFERENCE channels. A current-constant channel contributes0 and stays in the denominator. Entirely flat input abstains. No amplitude epsilon, channel selection, learned weights, new support or threshold.',
        controls='Apply this same one formula to8T4/T5, all34 standard decoder-input activity maps, and all five already defined8/34-channel input controls. Same old common support and19 shifts. Read all old score grids/maxima unchanged.',
        attribution='Only the two requested pairs: additive cell/channel accounting at the old winning shift and the new winning shift. Check sums against stored old scores within1e-9. GT box coordinates label locations only; they never enter the reader or support.',
        reuse='No model execution, image rendering, training or old score-grid recomputation. Evaluate only the new formula. Reconstructing additive terms at the two traced old/new locations is new attribution; its sum is checked against ready scores as an accounting identity.',
        populations='All200 existing fit20 requests including154unknowns and unsupported cases; preserve45supported known conditional labels. Additional transfer check on all seven earlier cases, only8-channel representations with already saved maps. These historical cases are not a new independent final test.',
        criteria=[dict(criterion='Mathematics, source hashes, same support/shifts, accounting sums1e-9 and unchanged ready results pass.',met=None),dict(criterion='On the two diagnostic pairs the new maxima order own>foreign for both C8 and C34 without parameter selection.',met=None),dict(criterion='On45matched known fit20 pairs neither FlyVis representation AUROC worsens; on the seven historical cases C8 AUROC does not worsen. Paired video-bootstrap intervals and all matched input controls must be reported; passing this point gate alone does not accept a tracker.',met=None)],
        stop='If the local or broader check fails, retain the explicit reader as a rejected experimental option; do not choose a channel mask, amplitude cutoff or another normalization from these two examples and do not deploy it. Strongest alternative is amplification of weak/noisy channels or instance information absent from this fixed correlation representation.',
        source=dict(title='Fast Normalized Cross-Correlation',author='J. P. Lewis',year=1995,url='https://scribblethink.org/Work/nvisionInterface/nip.html',role='Section2 Eq2 establishes normalized cross-correlation; section4 phase-correlation discussion explicitly notes the danger of equally weighting insignificant components. Per-neural-channel balancing is our tested extension, not a claimed published object-ID method.'),
        sources={f.relative_to(ROOT).as_posix():sha(f) for f in sorted(set(files))})
    write('protocol.json',p)
    path=ROOT/'experiments/project_history.json';cat=read(path);assert not any(c['id']==OUT.name for c in cat['experiments'])
    cat['experiments'].append(dict(id=OUT.name,title='Вклад ячеек и каналов — одинаковая нормировка пространственного сходства',area='flyvis',state='running',started='2026-11-09',question='Какие пространственные вклады дают чужой машине больший балл и помогает ли равный вес каналов?',hypothesis='Общий знаменатель может чрезмерно взвешивать отдельные каналы; независимая нормировка может убрать этот вклад, но усилить шум.',expected='Точный разбор двух пар и один вариант считывателя на прежних200событиях и семи дополнительных исторических случаях.',acceptance=p['criteria'],evidence=[dict(kind='file',ref=f'runs/{OUT.name}/protocol.json',claim='Условия до разложения естественных карт и новых оценок')]))
    path.write_text(json.dumps(cat,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    journal('Математика проверена; зарегистрирована одна нормировка каналов',f'*Постановка (`{OUT.name}`).* Сумма вкладов ячеек и каналов должна точно воспроизвести уже готовую оценку; разность двух пар раскладывается на изменение пространственного совпадения и относительного веса канала. Пять математических тестов выполнены до естественных карт. Один заранее заданный вариант — одинаковый вес нормированных по пространству каналов; без порога энергии или отбора типов. Те же правила для FlyVis и пяти контролей входа. Все200fit20запросов и семь старых случаев проверяются без повтора готовых оценок. Источник `runs/{OUT.name}/protocol.json`.')
    print('Registered one correction before natural-map attribution',flush=True)
def arrays(k,nb,wide=True,path=None):
    with np.load(path or BASE/'features'/(k+'.npz'),allow_pickle=False) as z:c8=z['C'].astype(float);t=z['R1'].astype(float)
    central=t[:,SELECT[:4]]
    out=dict(FlyVis=c8,center8=np.concatenate([central,central],1),neighbor8=neighbor_aug(central.T[None],nb)[0].T,input8_temporal=t[:,SELECT])
    if wide:
        with np.load(FULL/'maps'/(k+'.npz'),allow_pickle=False) as z:extra=z['C_OTHER']
        full=np.empty((721,34));full[:,4:12]=c8;full[:,OTHER]=extra
        out.update(C34=full,R134_temporal=t,R134_spatial=neighbor_aug(t[:,:17].T[None],nb)[0].T)
    return out
def explain(e,name,old,current,patches,baseline,values,g,truth):
    idx=np.asarray(e['geometry']['cells']);q=np.asarray(e['geometry']['query']);h=e['row_data']['history'][-1]
    c0=np.asarray(h['crop']);ct=np.asarray(e['row_data']['current_crop']);s0=(c0[2:]-c0[:2])/g['eye_size'];st=(ct[2:]-ct[:2])/g['eye_size']
    refxy=np.asarray(g['points'])[idx]*s0+c0[:2]
    refgt=next(o['box'] for o in truth['reference_history'][-1]['overlaps'] if o['identity']==truth['reference_history'][-1]['identity'])
    curgt=next(o['box'] for o in truth['current']['overlaps'] if o['identity']==truth['current']['identity'])
    oi=max((i for i,v in enumerate(baseline) if v is not None),key=lambda i:baseline[i]);ni=max((i for i,v in enumerate(values) if v is not None),key=lambda i:values[i])
    explanations={}
    for role,j in [('old_winner',oi),('new_winner',ni)]:
        if j==oi and role=='new_winner':explanations[role]=explanations['old_winner'];continue
        d=centered_contributions(old[idx],patches[j]);assert abs(d['score']-baseline[j])<=1e-9
        assert abs(d['balanced_score']-values[j])<=1e-12
        curxy=q[:,j]*st+ct[:2];cells=[]
        for k,cell in enumerate(idx):
            ra=np.asarray(refgt);ca=np.asarray(curgt)
            cells.append(dict(id=int(cell),reference_xy=refxy[k],current_xy=curxy[k],reference_box_relative=(refxy[k]-ra[:2])/(ra[2:]-ra[:2]),current_box_relative=(curxy[k]-ca[:2])/(ca[2:]-ca[:2]),
                reference_inside_annotation=bool(((refxy[k]>=ra[:2])&(refxy[k]<=ra[2:])).all()),current_inside_annotation=bool(((curxy[k]>=ca[:2])&(curxy[k]<=ca[2:])).all()),
                contribution=d['cell'][k],balanced_contribution=d['balanced_cell'][k],reference_energy=d['reference_cell_energy'][k],current_energy=d['current_cell_energy'][k]))
        explanations[role]=dict(index=j,offset=e['geometry']['offsets'][j],contributions=d,cells=cells)
    return dict(channels=channels(name),reference_annotation_box=refgt,current_annotation_box=curgt,reference_state_box=h['box'],current_detection_box=e['box'],explanations=explanations,
        limitation='Annotation rectangles localize cell centers only. Both boxes can contain background/occluders and neural receptive fields extend beyond a cell center. Additive terms do not equal the effect of removing that cell or neuron.')
def run():
    if (OUT/'results.json').exists():raise SystemExit('Ready new-reader results exist; no repetition.')
    p=read(OUT/'protocol.json');verify(p);g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');mesh=Delaunay(g['points']);nb=HexGeometry(np.array(g['points']),g['eye_size']).neighbors;sample=sampler()
    truths={key(x):x for x in read(ROOT/'runs/template_identity_audit_20261108/results.json')['records']};files={};attribution=[]
    for number,e in enumerate(read(PARENT/'protocol.json')['events']):
        dest=Path('events')/filename(e)
        if (OUT/dest).exists():files[dest.as_posix()]=sha(OUT/dest);continue
        a=read(PARENT/'events'/filename(e));b=read(WIDE/'events'/filename(e));idx=np.asarray(e['geometry']['cells']);variants={}
        if a['supported']:
            old=arrays(e['reference_key'],nb);current=arrays(e['current_key'],nb);query=np.asarray(e['geometry']['query'])
            for name in NAMES:
                prior=(a if name in a['variants'] else b)['variants'][name];baseline=prior['correlation_scores'];patches=sample(mesh,query,current[name]).transpose(1,0,2)
                values=[channel_balanced_similarity(old[name][idx],patch) for patch in patches];valid=[i for i,v in enumerate(values) if v is not None];win=max(valid,key=lambda i:values[i]) if valid else None
                variants[name]=dict(scores=values,maximum=values[win] if win is not None else None,winner=win,baseline_max=prior['correlation_max'],baseline_winner=max((i for i,v in enumerate(baseline) if v is not None),key=lambda i:baseline[i]) if any(v is not None for v in baseline) else None)
                if key(e) in TARGETS:
                    attribution.append(dict(video=e['video'],frame=e['frame'],old=e['old_tid'],new=e['new_tid'],representation=name,detail=explain(e,name,old[name],current[name],patches,baseline,values,g,truths[key(e)])))
        write(dest,dict(video=e['video'],frame=e['frame'],old=e['old_tid'],new=e['new_tid'],kind=a['kind'],supported=a['supported'],cells=a['cells'],variants=variants,protocol_sha256=sha(OUT/'protocol.json')));files[dest.as_posix()]=sha(OUT/dest)
        if number%20==0:print('New reader',number+1,'/',200,flush=True)
    assert len(attribution)==14
    differences={}
    for name in NAMES:
        own=next(x for x in attribution if x['old']==88 and x['representation']==name)['detail'];foreign=next(x for x in attribution if x['old']==73 and x['representation']==name)['detail']
        differences[name]=dict(channels=own['channels'],delta=channel_difference_attribution(own['explanations']['old_winner']['contributions'],foreign['explanations']['old_winner']['contributions']))
    write('attribution.json',dict(records=attribution,foreign_minus_own=differences,posthoc_scope='Two explicitly requested diagnostic pairs. Equal-weight correction was fixed before opening natural-map decomposition.'))
    transfer=[];oldrows=read(SEVEN/'results.json')['records']
    for c in read(SEVEN/'protocol.json')['cases']:
        t=c['geometry'];idx=np.asarray(t['eligible_cells']);c0=np.asarray(t['crop0']);ct=np.asarray(t['crop']);s0=np.asarray(t['scale0']);st=np.asarray(t['scale']);off=np.asarray(t['offsets']);d=np.asarray(t['prediction']);x=np.asarray(g['points'])[idx]*s0+c0[:2];query=(x[:,None]+d+off-ct[:2])/st
        both=[]
        for k in c['map_keys']:
            path=c['ready_maps'].get(k,f'runs/singleton_forward_backward_transfer_20261108/{k}.npz');both.append(arrays(k,nb,wide=False,path=ROOT/path))
        for name in NAMES[:4]:
            baseline=next(r for r in oldrows if (r['video'],r['frame'],r['old'],r['new'],r['representation'])==(c['video'],c['frame'],c['old'],c['new'],name))
            patches=sample(mesh,query,both[1][name]).transpose(1,0,2);values=[channel_balanced_similarity(both[0][name][idx],v) for v in patches];valid=[v for v in values if v is not None]
            transfer.append(dict(video=c['video'],frame=c['frame'],old=c['old'],new=c['new'],kind=c['kind'],representation=name,cells=idx.tolist(),scores=values,maximum=max(valid) if valid else None,baseline_max=baseline['forward_score']))
    write('transfer.json',dict(records=transfer,old_scores_read_only=True));verify(p)
    write('results.json',dict(protocol_sha256=sha(OUT/'protocol.json'),files=files,attribution_sha256=sha(OUT/'attribution.json'),transfer_sha256=sha(OUT/'transfer.json')))
    write('audit.json',dict(passed=True,source_hashes_stable=True,old_score_grids_reused=True,new_model_calls=0,new_image_reads=0,new_training=0,attribution_checks_old_score_sum_tolerance=1e-9,all200_events_preserved=True,seven_transfer_cases_preserved=True))
{'register':register,'run':run}[sys.argv[1]]()
